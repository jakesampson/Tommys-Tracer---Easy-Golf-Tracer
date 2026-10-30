use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs,
    io::{BufRead, BufReader, BufWriter, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::{GrayImage, Luma, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use imageproc::filter::box_filter;
use serde::{Deserialize, Serialize};

const EXPORT_FPS: u32 = 30;
const FRAME_LIMIT: u32 = 60 * 60 * 30;
const PREVIEW_MAX_DIMENSION: u32 = 720;
const PREVIEW_FRAME_MAX_DIMENSION: u32 = 1280;
const PLAYBACK_BUFFER_SECONDS: f32 = 0.5;
const PREVIEW_CACHE_BUDGET_BYTES: usize = 1024 * 1024 * 1024;
// How far ahead of the background decoder a requested frame may be before the decoder restarts there.
const PREVIEW_FILL_LOOKAHEAD_FRAMES: u64 = 45;
const PREVIEW_DECODE_AHEAD_FRAMES: u64 = 6;
const MAX_CANVAS_ZOOM: f32 = 8.0;
const TRACER_REFERENCE_HEIGHT: f32 = 1080.0;
const TRACER_WIDTH_MAX: f32 = 16.0;
const TRACER_GLOW_MAX: f32 = 24.0;
const STYLE_SCALE_MAX: i32 = 10;
const MOTION_ANALYSIS_MAX_FPS: f32 = 30.0;
const TRAIL_PACES: [f32; 3] = [0.25, 0.5, 1.0];

// Analyzing at the source frame rate (up to 30 fps) keeps per-frame camera movement small.
fn motion_analysis_fps(frame_rate: f32) -> f32 {
    if frame_rate.is_finite() && frame_rate > 0.0 {
        frame_rate.min(MOTION_ANALYSIS_MAX_FPS)
    } else {
        MOTION_ANALYSIS_MAX_FPS
    }
}

fn expected_motion_analysis_samples(duration: f32, analysis_fps: f32) -> usize {
    (duration * analysis_fps).ceil().max(1.0) as usize
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Keyframe {
    time: f32,
    x: f32,
    y: f32,
}

// A polygon in normalized video coordinates; the tracer is hidden inside it.
// With keyframes, each vertex moves linearly between them (all shapes share one vertex count).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MaskRegion {
    points: Vec<(f32, f32)>,
    #[serde(default)]
    keyframes: Vec<MaskKeyframe>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MaskKeyframe {
    time: f32,
    points: Vec<(f32, f32)>,
}

impl MaskRegion {
    fn new(points: Vec<(f32, f32)>) -> Self {
        Self {
            points,
            keyframes: Vec::new(),
        }
    }

    fn shape_at(&self, time: f32) -> Vec<(f32, f32)> {
        let (Some(first), Some(last)) = (self.keyframes.first(), self.keyframes.last()) else {
            return self.points.clone();
        };
        if time <= first.time {
            return first.points.clone();
        }
        if time >= last.time {
            return last.points.clone();
        }
        let next = self
            .keyframes
            .iter()
            .position(|keyframe| keyframe.time > time)
            .unwrap_or(self.keyframes.len() - 1);
        let (a, b) = (&self.keyframes[next - 1], &self.keyframes[next]);
        let amount = (time - a.time) / (b.time - a.time).max(f32::EPSILON);
        a.points
            .iter()
            .zip(&b.points)
            .map(|(p, q)| (p.0 + (q.0 - p.0) * amount, p.1 + (q.1 - p.1) * amount))
            .collect()
    }

    fn keyframe_index_at(&self, time: f32, frame_rate: f32) -> Option<usize> {
        self.keyframes
            .iter()
            .position(|keyframe| (keyframe.time - time).abs() < 0.5 / frame_rate)
    }

    fn ensure_keyframe(&mut self, time: f32, frame_rate: f32) -> usize {
        if let Some(index) = self.keyframe_index_at(time, frame_rate) {
            return index;
        }
        let keyframe = MaskKeyframe {
            time,
            points: self.shape_at(time),
        };
        let index = self
            .keyframes
            .iter()
            .position(|existing| existing.time > time)
            .unwrap_or(self.keyframes.len());
        self.keyframes.insert(index, keyframe);
        index
    }

    // Moves a vertex; an animated mask gets (or updates) a keyframe at this time.
    fn set_vertex(&mut self, vertex: usize, point: (f32, f32), time: f32, frame_rate: f32) {
        if self.keyframes.is_empty() {
            if let Some(slot) = self.points.get_mut(vertex) {
                *slot = point;
            }
            return;
        }
        let index = self.ensure_keyframe(time, frame_rate);
        if let Some(slot) = self.keyframes[index].points.get_mut(vertex) {
            *slot = point;
        }
    }

    // Adds a vertex after `edge_start` in every shape, placed at `point` for the current shape.
    fn insert_vertex(&mut self, edge_start: usize, point: (f32, f32), time: f32, frame_rate: f32) {
        let insert = |points: &mut Vec<(f32, f32)>| {
            let a = points[edge_start];
            let b = points[(edge_start + 1) % points.len()];
            points.insert(edge_start + 1, ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5));
        };
        if edge_start >= self.points.len() {
            return;
        }
        let current = (!self.keyframes.is_empty()).then(|| self.ensure_keyframe(time, frame_rate));
        insert(&mut self.points);
        for keyframe in &mut self.keyframes {
            insert(&mut keyframe.points);
        }
        match current {
            Some(index) => self.keyframes[index].points[edge_start + 1] = point,
            None => self.points[edge_start + 1] = point,
        }
    }

    // Moves the whole shape; an animated mask gets (or updates) a keyframe at this time.
    fn translate(&mut self, delta: (f32, f32), time: f32, frame_rate: f32) {
        let target = if self.keyframes.is_empty() {
            &mut self.points
        } else {
            let index = self.ensure_keyframe(time, frame_rate);
            &mut self.keyframes[index].points
        };
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for point in target.iter() {
            min_x = min_x.min(point.0);
            max_x = max_x.max(point.0);
            min_y = min_y.min(point.1);
            max_y = max_y.max(point.1);
        }
        let dx = delta.0.clamp(-min_x, 1.0 - max_x);
        let dy = delta.1.clamp(-min_y, 1.0 - max_y);
        for point in target.iter_mut() {
            point.0 += dx;
            point.1 += dy;
        }
    }

    fn remove_vertex(&mut self, vertex: usize) {
        if self.points.len() <= 3 || vertex >= self.points.len() {
            return;
        }
        self.points.remove(vertex);
        for keyframe in &mut self.keyframes {
            keyframe.points.remove(vertex);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Project {
    video_path: PathBuf,
    duration: f32,
    #[serde(default = "default_frame_rate")]
    frame_rate: f32,
    trim_in: f32,
    trim_out: f32,
    keyframes: Vec<Keyframe>,
    #[serde(default)]
    camera_motion: Vec<CameraMotionSample>,
    #[serde(default)]
    camera_motion_enabled: bool,
    #[serde(default)]
    masks: Vec<MaskRegion>,
    #[serde(default)]
    ball_colour: Option<[u8; 3]>,
    #[serde(default = "default_path_smoothing")]
    path_smoothing: f32,
    #[serde(default = "default_trail_duration")]
    trail_duration: f32,
    #[serde(default = "default_trail_pace")]
    trail_pace: f32,
    tracer: TracerStyle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CameraMotionSample {
    time: f32,
    x: f32,
    y: f32,
}

fn default_frame_rate() -> f32 {
    30.0
}

fn default_path_smoothing() -> f32 {
    0.5
}

fn default_trail_duration() -> f32 {
    5.0
}

fn default_trail_pace() -> f32 {
    0.5
}

fn trail_pace_index(pace: f32) -> i32 {
    if !pace.is_finite() {
        return 1;
    }
    TRAIL_PACES
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - pace).abs().total_cmp(&(*right - pace).abs()))
        .map_or(1, |(index, _)| index as i32)
}

fn trail_cutoff_time(first_time: f32, time: f32, duration: f32, pace: f32) -> f32 {
    let pace = TRAIL_PACES[trail_pace_index(pace) as usize];
    let elapsed_after_duration = (time - first_time - duration.max(0.0)).max(0.0);
    first_time + elapsed_after_duration * pace
}

fn trace_nudge_pixels(zoom_scale: f32, fine_nudge: bool) -> f32 {
    if fine_nudge {
        1.0
    } else if zoom_scale > 4.0 {
        5.0
    } else {
        10.0
    }
}

fn default_glow_color() -> [u8; 3] {
    [255, 76, 48]
}

fn style_scale_to_internal(value: i32, internal_max: f32) -> f32 {
    style_scale_to_internal_from(1, value, internal_max)
}

fn style_scale_to_internal_from(min_scale: i32, value: i32, internal_max: f32) -> f32 {
    value.clamp(min_scale, STYLE_SCALE_MAX) as f32 / STYLE_SCALE_MAX as f32 * internal_max
}

fn internal_to_style_scale(value: f32, internal_max: f32) -> i32 {
    internal_to_style_scale_from(1, value, internal_max)
}

fn internal_to_style_scale_from(min_scale: i32, value: f32, internal_max: f32) -> i32 {
    ((value / internal_max).clamp(0.0, 1.0) * STYLE_SCALE_MAX as f32)
        .round()
        .clamp(min_scale as f32, STYLE_SCALE_MAX as f32) as i32
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TracerStyle {
    color: [u8; 3],
    #[serde(default = "default_glow_color")]
    glow_color: [u8; 3],
    width: f32,
    glow: f32,
}

impl Default for TracerStyle {
    fn default() -> Self {
        Self {
            color: [255, 76, 48],
            glow_color: default_glow_color(),
            width: 5.0,
            glow: 10.0,
        }
    }
}

struct TracerApp {
    screen: AppScreen,
    project: Option<Project>,
    playhead_frame: u64,
    is_playing: bool,
    last_playback_tick: Instant,
    timeline_drag: Option<TimelineDrag>,
    preview_sender: Sender<PreviewCommand>,
    preview_receiver: mpsc::Receiver<PreviewResult>,
    preview_request_id: u64,
    displayed_preview_id: u64,
    preview: Option<egui::TextureHandle>,
    welcome_logo: egui::TextureHandle,
    tracer_glow: Option<egui::TextureHandle>,
    preview_size: [usize; 2],
    preview_rgba: Vec<u8>,
    canvas_zoom: CanvasZoom,
    motion_receiver: Option<Receiver<MotionAnalysisMessage>>,
    motion_cancel: Option<Arc<AtomicBool>>,
    trace_receiver: Option<Receiver<TraceMessage>>,
    trace_cancel: Option<Arc<AtomicBool>>,
    trace_progress: f32,
    keyframes_before_trace: Option<Vec<Keyframe>>,
    motion_analysis_selection: Option<(PathBuf, f32, f32)>,
    motion_analysis_frames: usize,
    motion_analysis_total: usize,
    motion_analysis_progress: f32,
    export_receiver: Option<Receiver<ExportMessage>>,
    export_progress: f32,
    export_phase: String,
    status: String,
}

struct CanvasZoom {
    scale: f32,
    pan: Vec2,
    dragging_keyframe: bool,
    keyframe_drag_offset: Vec2,
    keyboard_active: bool,
    mask_editing: bool,
    mask_draft: Vec<(f32, f32)>,
    selected_mask: Option<usize>,
    dragging_vertex: Option<usize>,
    dragging_shape: bool,
    picking_colour: bool,
    pick_request: Option<(f32, f32)>,
}

impl Default for CanvasZoom {
    fn default() -> Self {
        Self {
            scale: 1.0,
            pan: Vec2::ZERO,
            dragging_keyframe: false,
            keyframe_drag_offset: Vec2::ZERO,
            keyboard_active: false,
            mask_editing: false,
            mask_draft: Vec::new(),
            selected_mask: None,
            dragging_vertex: None,
            dragging_shape: false,
            picking_colour: false,
            pick_request: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppScreen {
    Welcome,
    Trim,
    Trace,
}

struct PreviewRequest {
    id: u64,
    path: PathBuf,
    time: f32,
    frame_rate: f32,
    // Inclusive frame range that should be kept cached (the trimmed clip).
    frame_range: (u64, u64),
}

enum PreviewCommand {
    Seek(PreviewRequest),
    Play(PreviewRequest),
    Pause,
}

struct PreviewResult {
    id: u64,
    time: f32,
    result: Result<PreviewFrame, String>,
    finished: bool,
}

#[derive(Default)]
struct PreviewFrame {
    size: [usize; 2],
    rgba: Vec<u8>,
}

enum ExportMessage {
    Progress { fraction: f32, phase: String },
    Finished(Result<PathBuf, String>),
}

enum MotionAnalysisMessage {
    Progress { completed: usize, total: usize },
    Finished(Result<Vec<CameraMotionSample>, String>),
}

enum TraceMessage {
    Progress(f32),
    Finished(Result<Vec<Keyframe>, String>),
}

#[derive(Clone, Copy)]
enum TimelineDrag {
    Playhead,
    TrimIn,
    TrimOut,
}

struct TimelineInteraction {
    refresh_preview: bool,
}

#[derive(Default)]
struct EditorInteraction {
    refresh_preview: bool,
    stop_playback: bool,
    toggle_playback: bool,
}

impl TracerApp {
    fn new(ctx: &egui::Context) -> Self {
        apply_app_theme(ctx);
        let logo_image =
            image::load_from_memory(include_bytes!("../assets/branding/welcome_logo.png"))
                .expect("embedded welcome logo must be a valid image")
                .to_rgba8();
        let welcome_logo = ctx.load_texture(
            "welcome-logo",
            egui::ColorImage::from_rgba_unmultiplied(
                [logo_image.width() as usize, logo_image.height() as usize],
                logo_image.as_raw(),
            ),
            egui::TextureOptions::LINEAR,
        );
        let (preview_sender, preview_requests) = mpsc::channel::<PreviewCommand>();
        let (preview_results, preview_receiver) = mpsc::sync_channel::<PreviewResult>(2);
        let repaint_context = ctx.clone();
        let status = match std::thread::Builder::new()
            .name("opentracer-preview".to_owned())
            .spawn(move || {
                preview_worker(preview_requests, preview_results, repaint_context);
            }) {
            Ok(_) => "Open a video to start a shot tracer project.".to_owned(),
            Err(error) => format!("Could not start video preview worker: {error}"),
        };
        Self {
            screen: AppScreen::Welcome,
            project: None,
            playhead_frame: 0,
            is_playing: false,
            last_playback_tick: Instant::now(),
            timeline_drag: None,
            preview_sender,
            preview_receiver,
            preview_request_id: 0,
            displayed_preview_id: 0,
            preview: None,
            welcome_logo,
            tracer_glow: None,
            preview_size: [0, 0],
            preview_rgba: Vec::new(),
            canvas_zoom: CanvasZoom::default(),
            motion_receiver: None,
            motion_cancel: None,
            trace_receiver: None,
            trace_cancel: None,
            trace_progress: 0.0,
            keyframes_before_trace: None,
            motion_analysis_selection: None,
            motion_analysis_frames: 0,
            motion_analysis_total: 0,
            motion_analysis_progress: 0.0,
            export_receiver: None,
            export_progress: 0.0,
            export_phase: String::new(),
            status,
        }
    }

    fn open_video(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Video", &["mp4", "mov", "mkv", "avi", "webm"])
            .pick_file()
        else {
            return;
        };

        match probe_video_info(&path) {
            Ok((duration, frame_rate)) if duration > 0.0 => {
                if self.is_playing {
                    if let Err(error) = self.preview_sender.send(PreviewCommand::Pause) {
                        self.status = format!("Could not stop previous video preview: {error}");
                    }
                }
                self.project = Some(Project {
                    video_path: path,
                    duration,
                    frame_rate,
                    trim_in: 0.0,
                    trim_out: duration,
                    keyframes: Vec::new(),
                    camera_motion: Vec::new(),
                    camera_motion_enabled: false,
                    masks: Vec::new(),
                    ball_colour: None,
                    path_smoothing: default_path_smoothing(),
                    trail_duration: default_trail_duration(),
                    trail_pace: default_trail_pace(),
                    tracer: TracerStyle::default(),
                });
                self.canvas_zoom = CanvasZoom::default();
                self.keyframes_before_trace = None;
                self.playhead_frame = 0;
                self.is_playing = false;
                self.screen = AppScreen::Trim;
                self.request_preview();
            }
            Ok(_) => self.status = "The selected video has no playable duration.".to_owned(),
            Err(error) => self.status = error,
        }
    }

    fn request_preview(&mut self) {
        let Some(project) = &self.project else {
            return;
        };
        let time = frame_time(self.playhead_frame, project.frame_rate);
        self.preview_request_id = self.preview_request_id.wrapping_add(1);
        if let Err(error) = self
            .preview_sender
            .send(PreviewCommand::Seek(PreviewRequest {
                id: self.preview_request_id,
                path: project.video_path.clone(),
                time,
                frame_rate: project.frame_rate,
                frame_range: trimmed_frame_range(project),
            }))
        {
            self.status = format!("Could not request video preview: {error}");
        }
    }

    fn start_playback(&mut self) {
        let Some(project) = &self.project else {
            return;
        };
        let start_frame = time_to_frame(project.trim_in, project.frame_rate);
        let end_frame =
            time_to_frame(project.trim_out, project.frame_rate).min(last_frame(project));
        if self.playhead_frame < start_frame || self.playhead_frame >= end_frame {
            self.playhead_frame = start_frame;
        }
        self.preview_request_id = self.preview_request_id.wrapping_add(1);
        self.displayed_preview_id = self.preview_request_id;
        if let Err(error) = self
            .preview_sender
            .send(PreviewCommand::Play(PreviewRequest {
                id: self.preview_request_id,
                path: project.video_path.clone(),
                time: frame_time(self.playhead_frame, project.frame_rate),
                frame_rate: project.frame_rate,
                frame_range: trimmed_frame_range(project),
            }))
        {
            self.status = format!("Could not start video preview: {error}");
            return;
        }
        self.last_playback_tick = Instant::now();
        self.is_playing = true;
    }

    fn pause_playback(&mut self) {
        self.is_playing = false;
        if let Err(error) = self.preview_sender.send(PreviewCommand::Pause) {
            self.status = format!("Could not pause video preview: {error}");
        }
    }

    fn save_project(&mut self) {
        let Some(project) = &mut self.project else {
            self.status = "Open a video before saving a project.".to_owned();
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Tommy's Tracer project", &["otracer"])
            .set_file_name("shot-tracer.otracer")
            .save_file()
        else {
            return;
        };
        match serde_json::to_vec_pretty(project)
            .map_err(|error| error.to_string())
            .and_then(|data| fs::write(&path, data).map_err(|error| error.to_string()))
        {
            Ok(()) => self.status = format!("Project saved to {}", path.display()),
            Err(error) => self.status = format!("Could not save project: {error}"),
        }
    }

    fn analyze_camera_motion(&mut self, ctx: &egui::Context) {
        let Some(project) = self.project.clone() else {
            self.status = "Open a video before analyzing camera motion.".to_owned();
            return;
        };
        if self.motion_receiver.is_some() {
            return;
        }

        let (sender, receiver) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let expected_frames = expected_motion_analysis_samples(
            project.trim_out - project.trim_in,
            motion_analysis_fps(project.frame_rate),
        );
        self.motion_analysis_selection = Some((
            project.video_path.clone(),
            project.trim_in,
            project.trim_out,
        ));
        let repaint_context = ctx.clone();
        match std::thread::Builder::new()
            .name("opentracer-motion-analysis".to_owned())
            .spawn(move || {
                let progress_sender = sender.clone();
                let result =
                    analyze_camera_motion(&project, &worker_cancelled, |completed, total| {
                        let _ = progress_sender
                            .send(MotionAnalysisMessage::Progress { completed, total });
                        repaint_context.request_repaint();
                    });
                let _ = sender.send(MotionAnalysisMessage::Finished(result));
                repaint_context.request_repaint();
            }) {
            Ok(_) => {
                self.motion_receiver = Some(receiver);
                self.motion_cancel = Some(cancelled);
                self.motion_analysis_frames = 0;
                self.motion_analysis_total = expected_frames;
                self.motion_analysis_progress = 0.0;
                self.status = "Stabilising footage...".to_owned();
            }
            Err(error) => {
                self.motion_analysis_selection = None;
                self.status = format!("Could not start camera-motion analysis: {error}");
            }
        }
    }

    fn start_ball_trace(&mut self, ctx: &egui::Context) {
        let Some(project) = self.project.clone() else {
            return;
        };
        if self.trace_receiver.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let repaint_context = ctx.clone();
        let spawned = std::thread::Builder::new()
            .name("opentracer-ball-trace".to_owned())
            .spawn(move || {
                let progress_sender = sender.clone();
                let progress_context = repaint_context.clone();
                let result = trace_ball(&project, &worker_cancelled, |completed, total| {
                    let _ = progress_sender.send(TraceMessage::Progress(
                        completed as f32 / total.max(1) as f32,
                    ));
                    progress_context.request_repaint();
                });
                let _ = sender.send(TraceMessage::Finished(result));
                repaint_context.request_repaint();
            });
        match spawned {
            Ok(_) => {
                self.trace_receiver = Some(receiver);
                self.trace_cancel = Some(cancelled);
                self.trace_progress = 0.0;
                self.status = "Tracing the ball...".to_owned();
            }
            Err(error) => self.status = format!("Could not start ball tracing: {error}"),
        }
    }

    fn cancel_ball_trace(&mut self) {
        if let Some(cancelled) = self.trace_cancel.take() {
            cancelled.store(true, Ordering::Relaxed);
        }
        self.trace_receiver = None;
    }

    fn poll_ball_trace(&mut self) {
        let mut finished = None;
        if let Some(receiver) = &self.trace_receiver {
            loop {
                match receiver.try_recv() {
                    Ok(TraceMessage::Progress(fraction)) => self.trace_progress = fraction,
                    Ok(TraceMessage::Finished(result)) => {
                        finished = Some(result);
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        finished = Some(Err("Ball tracing stopped unexpectedly.".to_owned()));
                        break;
                    }
                }
            }
        }
        let Some(result) = finished else {
            return;
        };
        self.trace_receiver = None;
        self.trace_cancel = None;
        match result {
            Ok(keyframes) => {
                if let Some(project) = &mut self.project {
                    self.keyframes_before_trace =
                        Some(std::mem::replace(&mut project.keyframes, keyframes));
                    self.status =
                        "Ball traced. Check the path and fix any keyframes that strayed, or undo."
                            .to_owned();
                }
            }
            Err(error) => self.status = format!("Ball tracing failed: {error}"),
        }
    }

    fn cancel_motion_analysis(&mut self) {
        if let Some(cancelled) = self.motion_cancel.take() {
            cancelled.store(true, Ordering::Relaxed);
        }
        self.motion_receiver = None;
        self.motion_analysis_selection = None;
    }

    fn load_project(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Tommy's Tracer project", &["otracer", "json"])
            .pick_file()
        else {
            return;
        };
        let result = fs::read(&path)
            .map_err(|error| error.to_string())
            .and_then(|data| {
                serde_json::from_slice::<Project>(&data).map_err(|error| error.to_string())
            });
        match result {
            Ok(mut project) => {
                if self.is_playing {
                    if let Err(error) = self.preview_sender.send(PreviewCommand::Pause) {
                        self.status = format!("Could not stop previous video preview: {error}");
                    }
                }
                if !project.frame_rate.is_finite() || project.frame_rate <= 0.0 {
                    project.frame_rate = default_frame_rate();
                }
                if !project.trail_duration.is_finite() {
                    project.trail_duration = default_trail_duration();
                }
                project.trail_duration = project.trail_duration.clamp(1.0, 10.0);
                project.trail_pace = TRAIL_PACES[trail_pace_index(project.trail_pace) as usize];
                project.duration = project.duration.max(0.0);
                project.trim_in = project.trim_in.clamp(0.0, project.duration);
                project.trim_out = project.trim_out.clamp(project.trim_in, project.duration);
                project.keyframes.sort_by(|a, b| a.time.total_cmp(&b.time));
                project.camera_motion.retain(|sample| {
                    sample.time.is_finite() && sample.x.is_finite() && sample.y.is_finite()
                });
                project
                    .camera_motion
                    .sort_by(|a, b| a.time.total_cmp(&b.time));
                project.camera_motion_enabled &= !project.camera_motion.is_empty();
                self.playhead_frame = time_to_frame(project.trim_in, project.frame_rate);
                self.project = Some(project);
                self.canvas_zoom = CanvasZoom::default();
                self.keyframes_before_trace = None;
                self.is_playing = false;
                self.screen = AppScreen::Trace;
                self.request_preview();
            }
            Err(error) => self.status = format!("Could not open project: {error}"),
        }
    }

    fn export_video(&mut self, ctx: &egui::Context) {
        let Some(project) = self.project.clone() else {
            self.status = "Open a video before exporting.".to_owned();
            return;
        };
        if project.trim_out <= project.trim_in {
            self.status = "Trim end must be later than trim start.".to_owned();
            return;
        }
        if project.keyframes.is_empty() {
            self.status = "Add at least one tracer keyframe before exporting.".to_owned();
            return;
        }
        let Some(output) = rfd::FileDialog::new()
            .add_filter("MP4 video", &["mp4"])
            .set_file_name("shot-tracer.mp4")
            .save_file()
        else {
            return;
        };
        let (sender, receiver) = mpsc::channel();
        let repaint_context = ctx.clone();
        match std::thread::Builder::new()
            .name("opentracer-export".to_owned())
            .spawn(move || {
                let progress_sender = sender.clone();
                let repaint_context = repaint_context.clone();
                let result = export_project(&project, &output, |fraction, phase| {
                    let _ = progress_sender.send(ExportMessage::Progress {
                        fraction,
                        phase: phase.to_owned(),
                    });
                    repaint_context.request_repaint();
                })
                .map(|()| output);
                let _ = sender.send(ExportMessage::Finished(result));
                repaint_context.request_repaint();
            }) {
            Ok(_) => {
                self.export_receiver = Some(receiver);
                self.export_progress = 0.0;
                self.export_phase = "Preparing export".to_owned();
                self.status = "Rendering video and tracer overlay...".to_owned();
            }
            Err(error) => self.status = format!("Could not start export: {error}"),
        }
    }

    fn toggle_playback(&mut self) {
        if self.project.is_none() {
            return;
        }
        if self.is_playing {
            self.pause_playback();
        } else {
            self.start_playback();
        }
    }

    fn show_welcome(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::from_rgb(19, 29, 41)))
            .show(ctx, |ui| {
                let ambient = ui.max_rect();
                ui.painter().circle_filled(
                    ambient.center() + Vec2::new(-180.0, -90.0),
                    250.0,
                    Color32::from_rgba_unmultiplied(47, 119, 213, 35),
                );
                ui.painter().circle_filled(
                    ambient.center() + Vec2::new(240.0, 140.0),
                    220.0,
                    Color32::from_rgba_unmultiplied(55, 191, 174, 22),
                );
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() * 0.13);
                    egui::Frame::new()
                        .fill(Color32::from_rgba_unmultiplied(28, 43, 60, 228))
                        .stroke(Stroke::new(1.0_f32, Color32::from_white_alpha(38)))
                        .corner_radius(egui::CornerRadius::same(24))
                        .inner_margin(egui::Margin::same(32))
                        .show(ui, |ui| {
                            ui.set_max_width(520.0);
                            ui.vertical_centered(|ui| {
                                ui.add(
                                    egui::Image::new((
                                        self.welcome_logo.id(),
                                        Vec2::new(420.0, 280.0),
                                    ))
                                    .fit_to_exact_size(Vec2::new(420.0, 280.0)),
                                );
                                ui.add_space(8.0);
                                ui.label(
                                egui::RichText::new(
                                    "Trace the ball flight in your golf videos, frame by frame.",
                                )
                                .color(Color32::from_rgb(218, 229, 241)),
                            );
                                ui.add_space(6.0);
                                ui.label(
                                egui::RichText::new(
                                    "Requires FFmpeg and FFprobe installed and available on PATH.",
                                )
                                .color(Color32::from_rgb(176, 195, 214))
                                .small(),
                            );
                                ui.add_space(24.0);
                                if ui
                                    .add_sized(
                                        [300.0, 42.0],
                                        egui::Button::new(
                                            egui::RichText::new("Select a video").strong(),
                                        )
                                        .fill(Color32::from_rgb(43, 132, 255)),
                                    )
                                    .clicked()
                                {
                                    self.open_video();
                                }
                                ui.add_space(8.0);
                                if ui
                                    .add_sized(
                                        [300.0, 38.0],
                                        egui::Button::new("Open a saved project").fill(
                                            Color32::from_rgba_unmultiplied(255, 255, 255, 16),
                                        ),
                                    )
                                    .clicked()
                                {
                                    self.load_project();
                                }
                            });
                        });
                });
            });
    }

    fn show_trim_screen(&mut self, ctx: &egui::Context) {
        let mut continue_to_trace = false;
        egui::TopBottomPanel::top("trim_toolbar")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(21, 32, 45, 245))
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Welcome").clicked() {
                        if self.is_playing {
                            self.pause_playback();
                        }
                        self.screen = AppScreen::Welcome;
                    }
                    ui.separator();
                    ui.label("1. Trim video");
                    if ui.button("Select another video").clicked() {
                        self.open_video();
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Continue to tracing").clicked() {
                            continue_to_trace = true;
                        }
                    });
                });
            });
        if continue_to_trace {
            self.continue_to_tracing(ctx);
            return;
        }
        let Some(project) = &mut self.project else {
            self.screen = AppScreen::Welcome;
            return;
        };
        let original_trim = (project.trim_in, project.trim_out);
        if self.screen == AppScreen::Trace {
            self.playhead_frame = time_to_frame(project.trim_in, project.frame_rate);
        }
        let mut interaction = EditorInteraction::default();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.label("Drag the green IN and OUT handles. The shaded section is kept.");
                interaction = draw_video_editor(
                    ui,
                    project,
                    &mut self.playhead_frame,
                    &mut self.is_playing,
                    &mut self.timeline_drag,
                    (
                        &self.preview,
                        self.preview_size,
                        &mut self.tracer_glow,
                        &mut self.canvas_zoom,
                    ),
                    true,
                );
            });
        });
        if original_trim != (project.trim_in, project.trim_out) {
            project.camera_motion.clear();
            project.camera_motion_enabled = false;
        }
        if self.screen == AppScreen::Trace {
            self.request_preview();
        }
        self.apply_editor_interaction(interaction);
    }

    fn continue_to_tracing(&mut self, ctx: &egui::Context) {
        if self.is_playing {
            self.pause_playback();
        }
        self.screen = AppScreen::Trace;
        let needs_analysis = self
            .project
            .as_ref()
            .is_some_and(|project| project.camera_motion.is_empty());
        if let Some(project) = &self.project {
            self.playhead_frame = time_to_frame(project.trim_in, project.frame_rate);
        }
        self.request_preview();
        if needs_analysis {
            self.analyze_camera_motion(ctx);
        }
    }

    fn show_stabilising_popup(&mut self, ctx: &egui::Context) {
        if self.motion_receiver.is_none() {
            return;
        }
        let mut skip = false;
        egui::Modal::new(egui::Id::new("stabilising_footage")).show(ctx, |ui| {
            ui.set_width(320.0);
            ui.vertical_centered(|ui| {
                ui.heading("Stabilising footage");
                ui.add_space(10.0);
                ui.add(
                    egui::ProgressBar::new(self.motion_analysis_progress)
                        .desired_width(280.0)
                        .show_percentage(),
                );
                ui.add_space(10.0);
                if ui.button("Skip").clicked() {
                    skip = true;
                }
            });
        });
        if skip {
            self.cancel_motion_analysis();
            self.status = "Stabilisation skipped.".to_owned();
        }
    }

    fn show_trace_screen(&mut self, ctx: &egui::Context) {
        let mut start_stabilise = false;
        egui::TopBottomPanel::top("trace_toolbar")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(21, 32, 45, 245))
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Back to trim").clicked() {
                        if self.is_playing {
                            self.pause_playback();
                        }
                        self.screen = AppScreen::Trim;
                    }
                    ui.separator();
                    ui.label("2. Trace ball flight");
                    if ui.button("Open project").clicked() {
                        self.load_project();
                    }
                    if ui.button("Save project").clicked() {
                        self.save_project();
                    }
                    if ui
                        .add_enabled(
                            self.project.is_some() && self.export_receiver.is_none(),
                            egui::Button::new("Export MP4"),
                        )
                        .clicked()
                    {
                        self.export_video(ctx);
                    }
                    if let Some(project) = &mut self.project {
                        let has_motion = !project.camera_motion.is_empty();
                        if has_motion {
                            ui.checkbox(&mut project.camera_motion_enabled, "Motion compensation");
                        }
                        if self.motion_receiver.is_none()
                            && ui
                                .button(if has_motion {
                                    "Re-stabilise"
                                } else {
                                    "Stabilise"
                                })
                                .on_hover_text(
                                    "Analyse camera movement so the tracer follows the scene.",
                                )
                                .clicked()
                        {
                            start_stabilise = true;
                        }
                    }
                });
            });
        if start_stabilise {
            self.analyze_camera_motion(ctx);
        }

        let mut start_trace = false;
        let mut cancel_trace = false;
        egui::SidePanel::right("trace_editor")
            .default_width(270.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(27, 40, 54, 242))
                    .inner_margin(egui::Margin::same(16)),
            )
            .show(ctx, |ui| {
                ui.heading("Tracer style");
                ui.add_space(8.0);
                if let Some(project) = &mut self.project {
                    let mut width_scale =
                        internal_to_style_scale(project.tracer.width, TRACER_WIDTH_MAX);
                    ui.horizontal(|ui| {
                        let slider_width = (ui.available_width()
                            - ui.spacing().interact_size.x
                            - ui.spacing().item_spacing.x)
                            .max(1.0);
                        if ui
                            .add_sized(
                                [slider_width, ui.spacing().interact_size.y],
                                egui::Slider::new(&mut width_scale, 1..=STYLE_SCALE_MAX)
                                    .text("Width"),
                            )
                            .on_hover_text("Style scale: 1–10, where 10 is 100%.")
                            .changed()
                        {
                            project.tracer.width =
                                style_scale_to_internal(width_scale, TRACER_WIDTH_MAX);
                        }
                        let mut color = Color32::from_rgb(
                            project.tracer.color[0],
                            project.tracer.color[1],
                            project.tracer.color[2],
                        );
                        if ui
                            .color_edit_button_srgba(&mut color)
                            .on_hover_text("Tracer line color")
                            .changed()
                        {
                            project.tracer.color = [color.r(), color.g(), color.b()];
                        }
                    });
                    let mut glow_scale =
                        internal_to_style_scale_from(0, project.tracer.glow, TRACER_GLOW_MAX);
                    ui.horizontal(|ui| {
                        let slider_width = (ui.available_width()
                            - ui.spacing().interact_size.x
                            - ui.spacing().item_spacing.x)
                            .max(1.0);
                        if ui
                            .add_sized(
                                [slider_width, ui.spacing().interact_size.y],
                                egui::Slider::new(&mut glow_scale, 0..=STYLE_SCALE_MAX)
                                    .text("Glow"),
                            )
                            .on_hover_text("Style scale: 0–10, where 0 turns the glow off and 10 is 100%.")
                            .changed()
                        {
                            project.tracer.glow =
                                style_scale_to_internal_from(0, glow_scale, TRACER_GLOW_MAX);
                        }
                        let mut color = Color32::from_rgb(
                            project.tracer.glow_color[0],
                            project.tracer.glow_color[1],
                            project.tracer.glow_color[2],
                        );
                        if ui
                            .color_edit_button_srgba(&mut color)
                            .on_hover_text("Glow color")
                            .changed()
                        {
                            project.tracer.glow_color = [color.r(), color.g(), color.b()];
                        }
                    });
                    let mut smoothing_scale =
                        internal_to_style_scale(project.path_smoothing, 1.0);
                    if ui
                        .add(
                            egui::Slider::new(&mut smoothing_scale, 1..=STYLE_SCALE_MAX)
                                .text("Smoothness"),
                        )
                        .on_hover_text(
                            "Style scale: 1–10, where 10 is 100% smoothing. Higher values round sharp turns.",
                        )
                        .changed()
                    {
                        project.path_smoothing =
                            style_scale_to_internal(smoothing_scale, 1.0);
                    }
                    ui.add(
                        egui::Slider::new(&mut project.trail_duration, 1.0..=10.0)
                            .step_by(1.0)
                            .text("Trail (seconds)"),
                    )
                    .on_hover_text(
                        "How long each part of the tracer remains visible, from 1 to 10 seconds.",
                    );
                    let mut trail_pace_index = trail_pace_index(project.trail_pace);
                    let trail_pace_label =
                        format!("Trail pace ({:.2}x)", TRAIL_PACES[trail_pace_index as usize]);
                    ui.add(
                        egui::Slider::new(&mut trail_pace_index, 0..=2)
                            .step_by(1.0)
                            .text(trail_pace_label),
                    )
                    .on_hover_text(
                        "After the selected trail duration, its old end moves forward at 0.25x, 0.5x, or 1.0x speed. Lower values erase the trail more slowly.",
                    );
                    project.trail_pace = TRAIL_PACES[trail_pace_index as usize];
                    ui.separator();
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.heading("Masks");
                        if !project.masks.is_empty()
                            && ui
                                .small_button(
                                    egui::RichText::new("Clear")
                                        .color(Color32::from_rgb(255, 151, 151)),
                                )
                                .clicked()
                        {
                            project.masks.clear();
                            self.canvas_zoom.selected_mask = None;
                        }
                    });
                    if self.canvas_zoom.mask_editing {
                        ui.label("Click the preview to add corners.")
                            .on_hover_text("Click the first point (or Finish) to close the shape.");
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    self.canvas_zoom.mask_draft.len() >= 3,
                                    egui::Button::new("Finish"),
                                )
                                .clicked()
                            {
                                project.masks.push(MaskRegion::new(std::mem::take(
                                    &mut self.canvas_zoom.mask_draft,
                                )));
                                self.canvas_zoom.selected_mask = Some(project.masks.len() - 1);
                                self.canvas_zoom.mask_editing = false;
                            }
                            if ui.button("Undo point").clicked() {
                                self.canvas_zoom.mask_draft.pop();
                            }
                            if ui.button("Cancel").clicked() {
                                self.canvas_zoom.mask_draft.clear();
                                self.canvas_zoom.mask_editing = false;
                            }
                        });
                    } else if ui
                        .button("Draw mask")
                        .on_hover_text(
                            "Outline an object the ball passes behind. The tracer is hidden inside the shape in the preview and the export.",
                        )
                        .clicked()
                    {
                        self.canvas_zoom.mask_draft.clear();
                        self.canvas_zoom.selected_mask = None;
                        self.canvas_zoom.mask_editing = true;
                    }
                    let mut remove_mask = None;
                    for index in 0..project.masks.len() {
                        ui.horizontal(|ui| {
                            let selected = self.canvas_zoom.selected_mask == Some(index);
                            let label = format!(
                                "Mask {}{}",
                                index + 1,
                                if project.masks[index].keyframes.is_empty() {
                                    String::new()
                                } else {
                                    format!(" ({} keyframes)", project.masks[index].keyframes.len())
                                }
                            );
                            if ui.selectable_label(selected, label).clicked() && !selected {
                                self.canvas_zoom.selected_mask = Some(index);
                                self.canvas_zoom.mask_editing = false;
                                self.canvas_zoom.mask_draft.clear();
                            } else if ui.small_button("x").clicked() {
                                remove_mask = Some(index);
                            }
                        });
                    }
                    if let Some(index) = remove_mask {
                        project.masks.remove(index);
                        self.canvas_zoom.selected_mask = match self.canvas_zoom.selected_mask {
                            Some(selected) if selected == index => None,
                            Some(selected) if selected > index => Some(selected - 1),
                            other => other,
                        };
                    }
                    if let Some(region) = self
                        .canvas_zoom
                        .selected_mask
                        .and_then(|index| project.masks.get_mut(index))
                    {
                        let time = frame_time(self.playhead_frame, project.frame_rate);
                        ui.horizontal(|ui| {
                            if ui
                                .button("Keyframe at playhead")
                                .on_hover_text(
                                    "Drag corners to reshape. Click an edge to add a corner; Shift+click a corner to delete it. The shape morphs between keyframes.",
                                )
                                .clicked()
                            {
                                region.ensure_keyframe(time, project.frame_rate);
                            }
                            if ui.button("Done").clicked() {
                                self.canvas_zoom.selected_mask = None;
                            }
                        });
                        let mut remove_keyframe = None;
                        if !region.keyframes.is_empty() {
                            egui::CollapsingHeader::new(format!(
                                "Mask keyframes ({})",
                                region.keyframes.len()
                            ))
                            .id_salt("mask_keyframes")
                            .show(ui, |ui| {
                                for (index, keyframe) in region.keyframes.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        ui.label(format!("{:.3}s", keyframe.time));
                                        if ui.small_button("x").clicked() {
                                            remove_keyframe = Some(index);
                                        }
                                    });
                                }
                            });
                        }
                        if let Some(index) = remove_keyframe {
                            if region.keyframes.len() == 1 {
                                region.points = region.keyframes[0].points.clone();
                            }
                            region.keyframes.remove(index);
                        }
                    }
                    ui.separator();
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.heading("Tracer");
                        if ui
                            .small_button(
                                egui::RichText::new("Clear")
                                    .color(Color32::from_rgb(255, 151, 151)),
                            )
                            .clicked()
                        {
                            project.keyframes.clear();
                        }
                    })
                    .response
                    .on_hover_text(
                        "Click the preview to add a keyframe. Arrow keys browse frames; WASD nudges the tracer anywhere in the tracing screen (10 preview pixels, 5 when zoomed past 400%, 1 with Shift).",
                    );
                    if self.trace_receiver.is_some() {
                        ui.add(egui::ProgressBar::new(self.trace_progress).text("Tracing ball"));
                        if ui.button("Cancel").clicked() {
                            cancel_trace = true;
                        }
                    } else {
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    project.keyframes.len() >= 2,
                                    egui::Button::new("Auto-trace ball"),
                                )
                                .on_hover_text(
                                    "Click the ball at its start, peak and end first. The app follows the ball between your first and last keyframes.",
                                )
                                .clicked()
                            {
                                start_trace = true;
                            }
                            if self.keyframes_before_trace.is_some()
                                && ui.button("Undo trace").clicked()
                            {
                                if let Some(previous) = self.keyframes_before_trace.take() {
                                    project.keyframes = previous;
                                }
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                            let picking = self.canvas_zoom.picking_colour;
                            if ui
                                .selectable_label(picking, "Pick ball colour")
                                .on_hover_text(
                                    "Click the ball in the preview so tracing can look for its colour. Zoom in first for a precise pick.",
                                )
                                .clicked()
                            {
                                self.canvas_zoom.picking_colour = !picking;
                                self.canvas_zoom.mask_editing = false;
                                self.canvas_zoom.selected_mask = None;
                            }
                        if let Some(colour) = project.ball_colour {
                            let (rect, _) = ui.allocate_exact_size(
                                Vec2::splat(18.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(
                                rect,
                                4.0,
                                Color32::from_rgb(colour[0], colour[1], colour[2]),
                            );
                            if ui.small_button("x").on_hover_text("Forget the picked colour").clicked()
                            {
                                project.ball_colour = None;
                            }
                        }
                    });
                    let mut remove_index = None;
                    egui::CollapsingHeader::new(format!("Keyframes ({})", project.keyframes.len()))
                        .id_salt("tracer_keyframes")
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical()
                                .max_height(ui.available_height())
                                .show(ui, |ui| {
                                    for (index, keyframe) in project.keyframes.iter().enumerate() {
                                        ui.horizontal(|ui| {
                                            let frame =
                                                time_to_frame(keyframe.time, project.frame_rate);
                                            if ui
                                                .selectable_label(
                                                    frame == self.playhead_frame,
                                                    format!(
                                                        "{:.3}s  ({:.2}, {:.2})",
                                                        keyframe.time, keyframe.x, keyframe.y
                                                    ),
                                                )
                                                .clicked()
                                            {
                                                self.playhead_frame = frame;
                                                self.is_playing = false;
                                            }
                                            if ui.small_button("x").clicked() {
                                                remove_index = Some(index);
                                            }
                                        });
                                    }
                                });
                        });
                    if let Some(index) = remove_index {
                        project.keyframes.remove(index);
                    }
                }
            });

        if start_trace {
            self.start_ball_trace(ctx);
        }
        if cancel_trace {
            self.cancel_ball_trace();
            self.status = "Ball tracing cancelled.".to_owned();
        }

        let Some(project) = &mut self.project else {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a video or project to start tracing.");
                    if ui.button("Open video").clicked() {
                        self.open_video();
                    }
                });
            });
            return;
        };
        let mut interaction = EditorInteraction::default();
        egui::CentralPanel::default().show(ctx, |ui| {
            interaction = draw_video_editor(
                ui,
                project,
                &mut self.playhead_frame,
                &mut self.is_playing,
                &mut self.timeline_drag,
                (
                    &self.preview,
                    self.preview_size,
                    &mut self.tracer_glow,
                    &mut self.canvas_zoom,
                ),
                false,
            );
        });
        if let Some(point) = self.canvas_zoom.pick_request.take() {
            self.status = match pick_ball_colour(&self.preview_rgba, self.preview_size, point) {
                Some(colour) => {
                    if let Some(project) = &mut self.project {
                        project.ball_colour = Some(colour);
                    }
                    "Ball colour picked. Auto-trace will look for it.".to_owned()
                }
                None => {
                    if let Some(project) = &mut self.project {
                        project.ball_colour = None;
                    }
                    "That spot is white or grey, so colour matching is off. Auto-trace will use brightness instead."
                        .to_owned()
                }
            };
        }
        self.apply_editor_interaction(interaction);
    }

    fn apply_editor_interaction(&mut self, interaction: EditorInteraction) {
        if interaction.toggle_playback {
            self.toggle_playback();
        } else if interaction.stop_playback && self.is_playing {
            self.pause_playback();
            if interaction.refresh_preview {
                self.request_preview();
            }
        } else if interaction.refresh_preview {
            self.request_preview();
        }
    }

    fn handle_trace_keyboard(&mut self, ctx: &egui::Context) {
        if self.screen != AppScreen::Trace {
            return;
        }

        let (horizontal, vertical, fine_nudge, step_backward, step_forward) =
            ctx.input_mut(|input| {
                let modifiers = input.modifiers;
                let horizontal = (if input.consume_key(modifiers, egui::Key::D) {
                    1.0
                } else {
                    0.0
                }) - (if input.consume_key(modifiers, egui::Key::A) {
                    1.0
                } else {
                    0.0
                });
                let vertical = (if input.consume_key(modifiers, egui::Key::S) {
                    1.0
                } else {
                    0.0
                }) - (if input.consume_key(modifiers, egui::Key::W) {
                    1.0
                } else {
                    0.0
                });
                let step_backward = input.consume_key(modifiers, egui::Key::ArrowLeft);
                let step_forward = input.consume_key(modifiers, egui::Key::ArrowRight);
                (
                    horizontal,
                    vertical,
                    modifiers.shift,
                    step_backward,
                    step_forward,
                )
            });
        if horizontal == 0.0 && vertical == 0.0 && !step_backward && !step_forward {
            return;
        }

        let next_frame = self.project.as_mut().and_then(|project| {
            let first_frame = time_to_frame(project.trim_in, project.frame_rate);
            let last_frame =
                time_to_frame(project.trim_out, project.frame_rate).min(last_frame(project));
            let direction = if step_forward { 1 } else { -1 };
            (step_backward || step_forward)
                .then(|| step_frame(self.playhead_frame, direction, first_frame, last_frame))
                .flatten()
        });
        if let Some(next_frame) = next_frame {
            if self.is_playing {
                self.pause_playback();
            }
            self.playhead_frame = next_frame;
            self.request_preview();
        }

        let nudge = Vec2::new(horizontal, vertical);
        if nudge != Vec2::ZERO {
            if let Some(project) = &mut self.project {
                let time = frame_time(self.playhead_frame, project.frame_rate);
                let Some(index) = ensure_keyframe_at_time(project, time) else {
                    return;
                };
                let pixels = trace_nudge_pixels(self.canvas_zoom.scale, fine_nudge);
                let width = self.preview_size[0].max(1) as f32;
                let height = self.preview_size[1].max(1) as f32;
                project.keyframes[index].x =
                    (project.keyframes[index].x + nudge.x * pixels / width).clamp(0.0, 1.0);
                project.keyframes[index].y =
                    (project.keyframes[index].y + nudge.y * pixels / height).clamp(0.0, 1.0);
            }
        }
    }
}

impl eframe::App for TracerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut latest_preview = None;
        while let Ok(preview_result) = self.preview_receiver.try_recv() {
            if preview_result.id < self.displayed_preview_id {
                continue;
            }
            self.displayed_preview_id = preview_result.id;
            if preview_result.finished {
                self.is_playing = false;
                continue;
            }
            latest_preview = Some(preview_result);
        }

        if let Some(preview_result) = latest_preview {
            let time = preview_result.time;
            if let Some(project) = &self.project {
                self.playhead_frame = time_to_frame(time, project.frame_rate);
                if self.is_playing && time >= project.trim_out {
                    self.pause_playback();
                }
            }

            match preview_result.result {
                Ok(frame) => {
                    let color_image =
                        egui::ColorImage::from_rgba_unmultiplied(frame.size, &frame.rgba);
                    if let Some(preview) = &mut self.preview {
                        preview.set(color_image, Default::default());
                    } else {
                        self.preview = Some(ctx.load_texture(
                            "video-preview",
                            color_image,
                            Default::default(),
                        ));
                    }
                    self.preview_size = frame.size;
                    self.preview_rgba = frame.rgba;
                    self.status = format!("Preview: frame {} ({time:.3}s)", self.playhead_frame);
                }
                Err(error) => self.status = error,
            }
        }

        self.handle_trace_keyboard(ctx);
        self.poll_ball_trace();

        let mut finished_motion = None;
        if let Some(receiver) = &self.motion_receiver {
            loop {
                match receiver.try_recv() {
                    Ok(MotionAnalysisMessage::Progress { completed, total }) => {
                        self.motion_analysis_frames = completed;
                        self.motion_analysis_total = total;
                        self.motion_analysis_progress =
                            (completed as f32 / total.max(1) as f32).clamp(0.0, 1.0);
                    }
                    Ok(MotionAnalysisMessage::Finished(result)) => {
                        finished_motion = Some(result);
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        finished_motion = Some(Err(
                            "Camera-motion analysis stopped unexpectedly.".to_owned()
                        ));
                        break;
                    }
                }
            }
        }
        if let Some(result) = finished_motion {
            self.motion_receiver = None;
            self.motion_cancel = None;
            let analyzed_selection = self.motion_analysis_selection.take();
            match result {
                Ok(samples) if samples.len() > 1 => {
                    let selection_matches = self.project.as_ref().is_some_and(|project| {
                        analyzed_selection
                            .as_ref()
                            .is_some_and(|(path, trim_in, trim_out)| {
                                path == &project.video_path
                                    && (trim_in - project.trim_in).abs() < f32::EPSILON
                                    && (trim_out - project.trim_out).abs() < f32::EPSILON
                            })
                    });
                    if selection_matches {
                        if let Some(project) = &mut self.project {
                            project.camera_motion = samples;
                            project.camera_motion_enabled = true;
                        }
                        self.motion_analysis_progress = 1.0;
                        self.status = "Footage stabilised.".to_owned();
                    } else {
                        self.status =
                            "Motion analysis finished, but the video trim changed; results were discarded."
                                .to_owned();
                    }
                }
                Ok(_) => self.status = "The trimmed clip is too short to analyze.".to_owned(),
                Err(error) => self.status = format!("Camera-motion analysis failed: {error}"),
            }
        }

        if (self.canvas_zoom.keyboard_active || !ctx.wants_keyboard_input())
            && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Space))
        {
            self.toggle_playback();
        }
        if self.is_playing {
            ctx.request_repaint_after(Duration::from_millis(12));
        }

        let mut finished_export = None;
        let mut export_disconnected = false;
        if let Some(receiver) = &self.export_receiver {
            loop {
                match receiver.try_recv() {
                    Ok(message) => match message {
                        ExportMessage::Progress { fraction, phase } => {
                            self.export_progress = fraction.clamp(0.0, 1.0);
                            self.export_phase = phase;
                        }
                        ExportMessage::Finished(result) => finished_export = Some(result),
                    },
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        export_disconnected = true;
                        break;
                    }
                }
            }
        }
        if let Some(result) = finished_export {
            match result {
                Ok(path) => {
                    self.export_progress = 1.0;
                    self.status = format!("Export complete: {}", path.display());
                }
                Err(error) => self.status = format!("Export failed: {error}"),
            }
            self.export_receiver = None;
            self.export_phase.clear();
        } else if export_disconnected {
            self.status = "Export stopped unexpectedly.".to_owned();
            self.export_receiver = None;
            self.export_phase.clear();
        }

        egui::TopBottomPanel::bottom("status")
            .frame(egui::Frame::new().fill(Color32::from_rgb(19, 29, 41)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(&self.status);
                    if self.export_receiver.is_some() {
                        ui.add(
                            egui::ProgressBar::new(self.export_progress)
                                .desired_width(200.0)
                                .text(format!(
                                    "{}: {:.0}%",
                                    self.export_phase,
                                    self.export_progress * 100.0
                                )),
                        );
                    }
                });
            });

        match self.screen {
            AppScreen::Welcome => self.show_welcome(ctx),
            AppScreen::Trim => self.show_trim_screen(ctx),
            AppScreen::Trace => self.show_trace_screen(ctx),
        }
        self.show_stabilising_popup(ctx);
    }
}

fn draw_video_editor(
    ui: &mut egui::Ui,
    project: &mut Project,
    playhead_frame: &mut u64,
    is_playing: &mut bool,
    timeline_drag: &mut Option<TimelineDrag>,
    preview_data: (
        &Option<egui::TextureHandle>,
        [usize; 2],
        &mut Option<egui::TextureHandle>,
        &mut CanvasZoom,
    ),
    trim_mode: bool,
) -> EditorInteraction {
    let mut interaction = EditorInteraction::default();
    let first_frame = if trim_mode {
        0
    } else {
        time_to_frame(project.trim_in, project.frame_rate)
    };
    let last_edit_frame = if trim_mode {
        last_frame(project)
    } else {
        time_to_frame(project.trim_out, project.frame_rate).min(last_frame(project))
    };
    *playhead_frame = (*playhead_frame).clamp(first_frame, last_edit_frame);
    let available = ui.available_size();
    let timeline_height = 78.0;
    let transport_height = 50.0;
    let video_height = (available.y - timeline_height - transport_height - 14.0).max(40.0);
    let (video_area, _) =
        ui.allocate_exact_size(Vec2::new(available.x, video_height), Sense::hover());
    ui.painter()
        .rect_filled(video_area, 12.0, Color32::from_rgb(9, 15, 23));
    ui.painter().rect_stroke(
        video_area,
        12.0,
        Stroke::new(1.0_f32, Color32::from_white_alpha(28)),
        egui::StrokeKind::Inside,
    );
    let base_preview_rect = if preview_data.1[0] > 0 && preview_data.1[1] > 0 {
        let aspect = preview_data.1[0] as f32 / preview_data.1[1] as f32;
        Some(Rect::from_center_size(
            video_area.center(),
            fit_size(video_area.size(), aspect),
        ))
    } else {
        None
    };
    let zoom_controls_rect = Rect::from_min_size(
        Pos2::new(video_area.right() - 166.0, video_area.top() + 10.0),
        Vec2::new(156.0, 34.0),
    );
    if let (Some(texture), Some(base_rect)) = (preview_data.0, base_preview_rect) {
        let zoom = preview_data.3;
        let image_rect = if trim_mode {
            base_rect
        } else {
            zoomed_preview_rect(video_area, base_rect.size(), zoom.scale, zoom.pan)
        };
        let canvas_response = ui.interact(
            video_area,
            ui.id().with("video-preview-canvas"),
            Sense::click_and_drag(),
        );
        let painter = ui.painter().with_clip_rect(video_area);
        painter.image(
            texture.id(),
            image_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        let mut zoom_changed = false;
        let controls_hovered = !trim_mode
            && canvas_response
                .interact_pointer_pos()
                .is_some_and(|pointer| zoom_controls_rect.contains(pointer));
        if !trim_mode {
            let pointer_clicked_outside_canvas = ui.input(|input| {
                input.pointer.any_click()
                    && input
                        .pointer
                        .interact_pos()
                        .is_some_and(|pointer| !video_area.contains(pointer))
            });
            if pointer_clicked_outside_canvas {
                zoom.keyboard_active = false;
            } else if canvas_response.clicked()
                || canvas_response.drag_started_by(egui::PointerButton::Primary)
                || canvas_response.drag_started_by(egui::PointerButton::Secondary)
            {
                zoom.keyboard_active = true;
            }
        }
        if !trim_mode {
            let panning = canvas_response.dragged_by(egui::PointerButton::Secondary);
            if panning {
                zoom.pan += canvas_response.drag_delta();
                zoom_changed = true;
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            }

            if !controls_hovered {
                if let Some(pointer) = canvas_response.hover_pos() {
                    if video_area.contains(pointer) {
                        let scroll_y = ui.input(|input| input.raw_scroll_delta.y);
                        if scroll_y != 0.0 {
                            let next_scale =
                                (zoom.scale * (1.0 + scroll_y * 0.002)).clamp(1.0, MAX_CANVAS_ZOOM);
                            if (next_scale - zoom.scale).abs() > f32::EPSILON {
                                zoom.pan = zoom_pan_for_cursor(
                                    video_area,
                                    image_rect,
                                    base_rect.size(),
                                    pointer,
                                    next_scale,
                                );
                                zoom.scale = next_scale;
                                zoom_changed = true;
                            }
                        }

                        if canvas_response.drag_started_by(egui::PointerButton::Primary)
                            && !zoom.mask_editing
                            && !zoom.picking_colour
                            && zoom.selected_mask.is_none()
                        {
                            if let Some(pointer) = canvas_response.interact_pointer_pos() {
                                let time = frame_time(*playhead_frame, project.frame_rate);
                                if let Some(index) = keyframe_index_at_time(
                                    &project.keyframes,
                                    time,
                                    project.frame_rate,
                                ) {
                                    let (offset_x, offset_y) = camera_motion_offset(project, time);
                                    let keyframe = &project.keyframes[index];
                                    let keyframe_position = Pos2::new(
                                        image_rect.left()
                                            + (keyframe.x + offset_x) * image_rect.width(),
                                        image_rect.top()
                                            + (keyframe.y + offset_y) * image_rect.height(),
                                    );
                                    if pointer.distance(keyframe_position) <= 18.0 {
                                        zoom.dragging_keyframe = true;
                                        zoom.keyframe_drag_offset = pointer - keyframe_position;
                                    }
                                }
                            }
                        }

                        if zoom.dragging_keyframe
                            && canvas_response.dragged_by(egui::PointerButton::Primary)
                        {
                            if let (Some(pointer), Some(index)) = (
                                canvas_response.interact_pointer_pos(),
                                keyframe_index_at_time(
                                    &project.keyframes,
                                    frame_time(*playhead_frame, project.frame_rate),
                                    project.frame_rate,
                                ),
                            ) {
                                let time = frame_time(*playhead_frame, project.frame_rate);
                                let (offset_x, offset_y) = camera_motion_offset(project, time);
                                let position = pointer - zoom.keyframe_drag_offset;
                                project.keyframes[index].x = ((position.x - image_rect.left())
                                    / image_rect.width()
                                    - offset_x)
                                    .clamp(0.0, 1.0);
                                project.keyframes[index].y = ((position.y - image_rect.top())
                                    / image_rect.height()
                                    - offset_y)
                                    .clamp(0.0, 1.0);
                            }
                        }
                        if canvas_response.drag_stopped_by(egui::PointerButton::Primary) {
                            zoom.dragging_keyframe = false;
                            zoom.keyframe_drag_offset = Vec2::ZERO;
                        }
                    }
                }
            }
        }

        if !trim_mode {
            draw_preview_tracer(
                ui,
                &painter,
                image_rect,
                project,
                frame_time(*playhead_frame, project.frame_rate),
                zoom.scale,
                preview_data.2,
            );
            let preview_time = frame_time(*playhead_frame, project.frame_rate);
            draw_preview_masks(
                &painter,
                image_rect,
                texture.id(),
                project,
                zoom,
                preview_time,
            );
            if canvas_response.clicked_by(egui::PointerButton::Primary)
                || canvas_response.drag_started_by(egui::PointerButton::Primary)
            {
                canvas_response.request_focus();
            }
            if zoom.selected_mask.is_some() && !zoom.mask_editing {
                let shift = ui.input(|input| input.modifiers.shift);
                edit_selected_mask(
                    &canvas_response,
                    image_rect,
                    project,
                    zoom,
                    preview_time,
                    shift,
                    controls_hovered,
                );
            }
            if zoom.picking_colour {
                if canvas_response.hovered() && !controls_hovered {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                }
                if !controls_hovered
                    && !canvas_response.dragged()
                    && canvas_response.clicked_by(egui::PointerButton::Primary)
                {
                    if let Some(position) = canvas_response.interact_pointer_pos() {
                        if image_rect.contains(position) {
                            zoom.pick_request = Some((
                                (position.x - image_rect.left()) / image_rect.width(),
                                (position.y - image_rect.top()) / image_rect.height(),
                            ));
                            zoom.picking_colour = false;
                        }
                    }
                }
            } else if zoom.mask_editing
                && !controls_hovered
                && !canvas_response.dragged()
                && canvas_response.clicked_by(egui::PointerButton::Primary)
            {
                if let Some(position) = canvas_response.interact_pointer_pos() {
                    if image_rect.contains(position) {
                        let point = (
                            ((position.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0),
                            ((position.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0),
                        );
                        let closes_shape = zoom.mask_draft.len() >= 3
                            && zoom.mask_draft.first().is_some_and(|first| {
                                Pos2::new(
                                    image_rect.left() + first.0 * image_rect.width(),
                                    image_rect.top() + first.1 * image_rect.height(),
                                )
                                .distance(position)
                                    <= 12.0
                            });
                        if closes_shape {
                            project
                                .masks
                                .push(MaskRegion::new(std::mem::take(&mut zoom.mask_draft)));
                            zoom.selected_mask = Some(project.masks.len() - 1);
                            zoom.mask_editing = false;
                        } else {
                            zoom.mask_draft.push(point);
                        }
                    }
                }
            } else if zoom.selected_mask.is_none()
                && !zoom.dragging_keyframe
                && !controls_hovered
                && !canvas_response.dragged()
                && canvas_response.clicked_by(egui::PointerButton::Primary)
            {
                if let Some(position) = canvas_response.interact_pointer_pos() {
                    if image_rect.contains(position) {
                        let time = frame_time(*playhead_frame, project.frame_rate);
                        let (offset_x, offset_y) = camera_motion_offset(project, time);
                        add_keyframe(
                            project,
                            time,
                            (((position.x - image_rect.left()) / image_rect.width()) - offset_x)
                                .clamp(0.0, 1.0),
                            (((position.y - image_rect.top()) / image_rect.height()) - offset_y)
                                .clamp(0.0, 1.0),
                        );
                    }
                }
            }
        }

        if !trim_mode {
            egui::Area::new(ui.id().with("zoom-controls"))
                .order(egui::Order::Foreground)
                .fixed_pos(zoom_controls_rect.min)
                .constrain_to(video_area)
                .show(ui.ctx(), |ui| {
                    egui::Frame::new()
                        .fill(Color32::from_rgba_unmultiplied(19, 29, 41, 225))
                        .stroke(Stroke::new(1.0_f32, Color32::from_white_alpha(36)))
                        .corner_radius(egui::CornerRadius::same(8))
                        .inner_margin(egui::Margin::symmetric(5, 3))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui.small_button("-").clicked() {
                                    zoom.scale = (zoom.scale - 0.25).max(1.0);
                                    if zoom.scale == 1.0 {
                                        zoom.pan = Vec2::ZERO;
                                    }
                                    zoom_changed = true;
                                }
                                ui.label(format!("{:.0}%", zoom.scale * 100.0));
                                if ui.small_button("+").clicked() {
                                    zoom.scale = (zoom.scale + 0.25).min(MAX_CANVAS_ZOOM);
                                    zoom_changed = true;
                                }
                                if ui.small_button("Reset").clicked() {
                                    zoom.scale = 1.0;
                                    zoom.pan = Vec2::ZERO;
                                    zoom_changed = true;
                                }
                            });
                        });
                });
        }

        if zoom_changed {
            zoom.pan = clamp_canvas_pan(video_area, base_rect.size() * zoom.scale, zoom.pan);
        }
    } else {
        ui.painter()
            .rect_filled(video_area, 12.0, Color32::from_rgb(9, 15, 23));
        ui.painter().rect_stroke(
            video_area,
            12.0,
            Stroke::new(1.0_f32, Color32::from_white_alpha(28)),
            egui::StrokeKind::Inside,
        );
        ui.painter().text(
            video_area.center(),
            egui::Align2::CENTER_CENTER,
            "Preparing video preview...",
            egui::FontId::proportional(16.0),
            Color32::GRAY,
        );
    }

    ui.add_space(6.0);
    let transport_card_width = 650.0_f32.min(ui.available_width());
    ui.horizontal(|ui| {
        ui.add_space(((ui.available_width() - transport_card_width) * 0.5).max(0.0));
        egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(37, 53, 71, 225))
            .stroke(Stroke::new(1.0_f32, Color32::from_white_alpha(28)))
            .corner_radius(egui::CornerRadius::same(12))
            .inner_margin(egui::Margin {
                left: 10,
                right: 10,
                top: 5,
                bottom: 5,
            })
            .show(ui, |ui| {
                ui.set_min_width((transport_card_width - 20.0).max(0.0));
                ui.horizontal(|ui| {
                    if ui.button("◀ Frame").clicked() {
                        interaction.stop_playback = *is_playing;
                        *playhead_frame = playhead_frame.saturating_sub(1).max(first_frame);
                        interaction.refresh_preview = true;
                    }
                    if ui
                        .add_sized(
                            [100.0, 30.0],
                            egui::Button::new(
                                egui::RichText::new(if *is_playing {
                                    "❚❚  Pause"
                                } else {
                                    "▶  Play"
                                })
                                .strong(),
                            )
                            .fill(if *is_playing {
                                Color32::from_rgb(56, 108, 163)
                            } else {
                                Color32::from_rgb(43, 132, 255)
                            }),
                        )
                        .clicked()
                    {
                        interaction.toggle_playback = true;
                    }
                    if ui.button("Frame ▶").clicked() {
                        interaction.stop_playback = *is_playing;
                        *playhead_frame = playhead_frame.saturating_add(1).min(last_edit_frame);
                        interaction.refresh_preview = true;
                    }
                    ui.separator();
                    let frame_response = ui.add(
                        egui::DragValue::new(playhead_frame)
                            .range(first_frame..=last_edit_frame)
                            .speed(1)
                            .prefix("Frame "),
                    );
                    if frame_response.changed() {
                        interaction.stop_playback = *is_playing;
                        *playhead_frame = (*playhead_frame).clamp(first_frame, last_edit_frame);
                        interaction.refresh_preview = true;
                    }
                    ui.label(format!(
                        "{:.3}s / {:.2} fps",
                        frame_time(*playhead_frame, project.frame_rate),
                        project.frame_rate
                    ));
                    if !trim_mode {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .button(
                                    egui::RichText::new("+  Add keyframe")
                                        .color(Color32::from_rgb(126, 190, 255)),
                                )
                                .clicked()
                            {
                                let time = frame_time(*playhead_frame, project.frame_rate);
                                let (offset_x, offset_y) = camera_motion_offset(project, time);
                                add_keyframe(
                                    project,
                                    time,
                                    (0.5 - offset_x).clamp(0.0, 1.0),
                                    (0.5 - offset_y).clamp(0.0, 1.0),
                                );
                            }
                        });
                    }
                });
            });
    });

    ui.add_space(2.0);
    let timeline_interaction = draw_timeline(ui, project, playhead_frame, timeline_drag, trim_mode);
    interaction.refresh_preview |= timeline_interaction.refresh_preview;
    if timeline_drag.is_some() {
        interaction.stop_playback |= *is_playing;
    }
    interaction
}

fn fit_size(available: Vec2, aspect: f32) -> Vec2 {
    let width = available.x.max(1.0);
    let height = available.y.max(1.0);
    if width / height > aspect {
        Vec2::new(height * aspect, height)
    } else {
        Vec2::new(width, width / aspect)
    }
}

fn zoomed_preview_rect(canvas: Rect, base_size: Vec2, scale: f32, pan: Vec2) -> Rect {
    let scale = scale.clamp(1.0, MAX_CANVAS_ZOOM);
    let size = base_size * scale;
    let pan = clamp_canvas_pan(canvas, size, pan);
    Rect::from_center_size(canvas.center() + pan, size)
}

fn clamp_canvas_pan(canvas: Rect, image_size: Vec2, pan: Vec2) -> Vec2 {
    let max_x = ((image_size.x - canvas.width()) * 0.5).max(0.0);
    let max_y = ((image_size.y - canvas.height()) * 0.5).max(0.0);
    Vec2::new(pan.x.clamp(-max_x, max_x), pan.y.clamp(-max_y, max_y))
}

fn zoom_pan_for_cursor(
    canvas: Rect,
    image: Rect,
    base_size: Vec2,
    pointer: Pos2,
    next_scale: f32,
) -> Vec2 {
    let image_uv = Vec2::new(
        (pointer.x - image.left()) / image.width(),
        (pointer.y - image.top()) / image.height(),
    );
    let next_size = base_size * next_scale;
    let pan = pointer - canvas.center() - Vec2::new(image_uv.x - 0.5, image_uv.y - 0.5) * next_size;
    clamp_canvas_pan(canvas, next_size, pan)
}

fn frame_time(frame: u64, frame_rate: f32) -> f32 {
    frame as f32 / frame_rate
}

fn time_to_frame(time: f32, frame_rate: f32) -> u64 {
    (time.max(0.0) * frame_rate).round() as u64
}

fn last_frame(project: &Project) -> u64 {
    (project.duration * project.frame_rate).ceil().max(1.0) as u64 - 1
}

fn trimmed_frame_range(project: &Project) -> (u64, u64) {
    let start = time_to_frame(project.trim_in, project.frame_rate).min(last_frame(project));
    let end = time_to_frame(project.trim_out, project.frame_rate).min(last_frame(project));
    (start, end.max(start))
}

fn playback_frame_deadline(started_at: Instant, frame_index: u64, frame_rate: f32) -> Instant {
    started_at + Duration::from_secs_f64(frame_index as f64 / f64::from(frame_rate))
}

fn timeline_range(duration: f32, trim_in: f32, trim_out: f32, trim_mode: bool) -> (f32, f32) {
    if trim_mode {
        (0.0, duration)
    } else {
        (trim_in, trim_out)
    }
}

fn snap_trim_time(time: f32, duration: f32, frame_rate: f32) -> f32 {
    if duration - time <= 0.5 / frame_rate {
        duration
    } else {
        (time * frame_rate).round() / frame_rate
    }
}

fn draw_timeline(
    ui: &mut egui::Ui,
    project: &mut Project,
    playhead_frame: &mut u64,
    drag_mode: &mut Option<TimelineDrag>,
    allow_trimming: bool,
) -> TimelineInteraction {
    let available_width = ui.available_width();
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(available_width, 72.0), Sense::click_and_drag());
    let painter = ui.painter();
    let (visible_start, visible_end) = timeline_range(
        project.duration,
        project.trim_in,
        project.trim_out,
        allow_trimming,
    );
    let duration = (visible_end - visible_start).max(f32::EPSILON);
    let left = rect.left() + 18.0;
    let right = rect.right() - 18.0;
    let track = Rect::from_min_max(
        Pos2::new(left, rect.top() + 30.0),
        Pos2::new(right.max(left + 1.0), rect.top() + 48.0),
    );
    let x_at_time = |time: f32| {
        track.left() + (time - visible_start).clamp(0.0, duration) / duration * track.width()
    };
    let time_at_x =
        |x: f32| visible_start + ((x - track.left()) / track.width()).clamp(0.0, 1.0) * duration;
    let trim_start_x = x_at_time(project.trim_in);
    let trim_end_x = x_at_time(project.trim_out);
    let playhead_time = frame_time(*playhead_frame, project.frame_rate);
    let playhead_x = x_at_time(playhead_time);
    let starting_trim_drag = allow_trimming
        && response.drag_started()
        && response.interact_pointer_pos().is_some_and(|pointer| {
            (pointer.x - trim_start_x)
                .abs()
                .min((pointer.x - trim_end_x).abs())
                <= 16.0
        });
    let active_trim_drag = allow_trimming
        && ((matches!(
            *drag_mode,
            Some(TimelineDrag::TrimIn | TimelineDrag::TrimOut)
        ) && !response.drag_stopped())
            || starting_trim_drag);
    if allow_trimming {
        let hovering_trim_handle = response.hovered()
            && response.interact_pointer_pos().is_some_and(|pointer| {
                (pointer.x - trim_start_x)
                    .abs()
                    .min((pointer.x - trim_end_x).abs())
                    <= 16.0
            });
        if active_trim_drag || hovering_trim_handle {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
    }

    painter.text(
        Pos2::new(track.left(), rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        if allow_trimming {
            "TIMELINE  |  Drag handles to trim; shaded area is kept"
        } else {
            "TIMELINE  |  Drag to move the playhead"
        },
        egui::FontId::proportional(11.0),
        Color32::GRAY,
    );
    painter.text(
        Pos2::new(track.right(), rect.top() + 4.0),
        egui::Align2::RIGHT_TOP,
        if allow_trimming {
            format!("{:.3}s - {:.3}s", project.trim_in, project.trim_out)
        } else {
            format!(
                "TRIMMED CLIP  {:.3}s - {:.3}s",
                project.trim_in, project.trim_out
            )
        },
        egui::FontId::monospace(11.0),
        Color32::LIGHT_GRAY,
    );
    if allow_trimming {
        painter.rect_filled(track, 3.0, Color32::from_rgb(43, 47, 52));
        painter.rect_filled(
            Rect::from_min_max(track.min, Pos2::new(trim_start_x, track.bottom())),
            3.0,
            Color32::from_rgba_unmultiplied(105, 48, 48, 150),
        );
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(trim_end_x, track.top()), track.max),
            3.0,
            Color32::from_rgba_unmultiplied(105, 48, 48, 150),
        );
    }
    painter.rect_filled(
        if allow_trimming {
            Rect::from_min_max(
                Pos2::new(trim_start_x, track.top()),
                Pos2::new(trim_end_x.max(trim_start_x + 1.0), track.bottom()),
            )
        } else {
            track
        },
        3.0,
        Color32::from_rgb(52, 94, 79),
    );

    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let x = track.left() + track.width() * fraction;
        painter.line_segment(
            [
                Pos2::new(x, track.bottom() + 2.0),
                Pos2::new(x, track.bottom() + 6.0),
            ],
            Stroke::new(1.0_f32, Color32::GRAY),
        );
        painter.text(
            Pos2::new(x, track.bottom() + 7.0),
            egui::Align2::CENTER_TOP,
            format!("{:.1}s", duration * fraction),
            egui::FontId::monospace(9.0),
            Color32::GRAY,
        );
    }

    for keyframe in project
        .keyframes
        .iter()
        .filter(|keyframe| keyframe.time >= visible_start && keyframe.time <= visible_end)
    {
        let x = x_at_time(keyframe.time);
        painter.circle_filled(
            Pos2::new(x, track.center().y),
            3.5,
            Color32::from_rgb(255, 219, 92),
        );
    }

    if allow_trimming {
        for (x, color) in [
            (trim_start_x, Color32::from_rgb(95, 211, 147)),
            (trim_end_x, Color32::from_rgb(95, 211, 147)),
        ] {
            painter.line_segment(
                [
                    Pos2::new(x, track.top() - 6.0),
                    Pos2::new(x, track.bottom() + 1.0),
                ],
                Stroke::new(2.0_f32, color),
            );
            painter.rect_filled(
                Rect::from_center_size(Pos2::new(x, track.center().y), Vec2::new(7.0, 24.0)),
                2.0,
                color,
            );
        }
        painter.text(
            Pos2::new(trim_start_x, track.top() - 8.0),
            egui::Align2::CENTER_BOTTOM,
            "IN",
            egui::FontId::monospace(9.0),
            Color32::from_rgb(95, 211, 147),
        );
        painter.text(
            Pos2::new(trim_end_x, track.top() - 8.0),
            egui::Align2::CENTER_BOTTOM,
            "OUT",
            egui::FontId::monospace(9.0),
            Color32::from_rgb(95, 211, 147),
        );
    }
    if !active_trim_drag {
        painter.line_segment(
            [
                Pos2::new(playhead_x, rect.top() + 20.0),
                Pos2::new(playhead_x, track.bottom() + 2.0),
            ],
            Stroke::new(2.0_f32, Color32::WHITE),
        );
        painter.circle_filled(
            Pos2::new(playhead_x, rect.top() + 20.0),
            4.0,
            Color32::WHITE,
        );
    }

    if response.drag_started() {
        if let Some(pointer) = response.interact_pointer_pos() {
            let start_distance = (pointer.x - trim_start_x).abs();
            let end_distance = (pointer.x - trim_end_x).abs();
            *drag_mode = Some(
                if allow_trimming && start_distance.min(end_distance) <= 16.0 {
                    if start_distance <= end_distance {
                        TimelineDrag::TrimIn
                    } else {
                        TimelineDrag::TrimOut
                    }
                } else {
                    TimelineDrag::Playhead
                },
            );
        }
    }

    if response.dragged() {
        if let (Some(mode), Some(pointer)) = (*drag_mode, response.interact_pointer_pos()) {
            let time = time_at_x(pointer.x);
            let one_frame = 1.0 / project.frame_rate;
            match mode {
                TimelineDrag::Playhead => {
                    let first_frame = time_to_frame(visible_start, project.frame_rate);
                    let last_frame =
                        time_to_frame(visible_end, project.frame_rate).min(last_frame(project));
                    *playhead_frame =
                        time_to_frame(time, project.frame_rate).clamp(first_frame, last_frame);
                }
                TimelineDrag::TrimIn => {
                    project.trim_in = snap_trim_time(time, duration, project.frame_rate)
                        .min((project.trim_out - one_frame).max(0.0));
                    *playhead_frame = time_to_frame(project.trim_in, project.frame_rate);
                }
                TimelineDrag::TrimOut => {
                    project.trim_out = snap_trim_time(time, duration, project.frame_rate)
                        .max((project.trim_in + one_frame).min(duration));
                    *playhead_frame = time_to_frame(project.trim_out, project.frame_rate)
                        .min(last_frame(project));
                }
            }
        }
    }

    if response.clicked() {
        if let Some(pointer) = response.interact_pointer_pos() {
            let first_frame = time_to_frame(visible_start, project.frame_rate);
            let last_frame =
                time_to_frame(visible_end, project.frame_rate).min(last_frame(project));
            *playhead_frame = time_to_frame(time_at_x(pointer.x), project.frame_rate)
                .clamp(first_frame, last_frame);
        }
    }

    let refresh_preview = response.clicked() || response.dragged() || response.drag_stopped();
    if response.drag_stopped() {
        *drag_mode = None;
    }
    TimelineInteraction { refresh_preview }
}

fn add_keyframe(project: &mut Project, time: f32, x: f32, y: f32) {
    project
        .keyframes
        .retain(|key| (key.time - time).abs() > 0.001);
    project.keyframes.push(Keyframe {
        time: time.clamp(0.0, project.duration),
        x,
        y,
    });
    project.keyframes.sort_by(|a, b| a.time.total_cmp(&b.time));
}

fn keyframe_index_at_time(keyframes: &[Keyframe], time: f32, frame_rate: f32) -> Option<usize> {
    let tolerance = 0.5 / frame_rate.max(1.0);
    keyframes
        .iter()
        .enumerate()
        .min_by(|(_, first), (_, second)| {
            (first.time - time)
                .abs()
                .total_cmp(&(second.time - time).abs())
        })
        .filter(|(_, keyframe)| (keyframe.time - time).abs() <= tolerance)
        .map(|(index, _)| index)
}

fn step_frame(current_frame: u64, direction: i8, first_frame: u64, last_frame: u64) -> Option<u64> {
    let target_frame = match direction.cmp(&0) {
        std::cmp::Ordering::Less => current_frame.checked_sub(1)?.max(first_frame),
        std::cmp::Ordering::Greater => current_frame.saturating_add(1).min(last_frame),
        std::cmp::Ordering::Equal => return None,
    };
    (target_frame != current_frame).then_some(target_frame)
}

fn ensure_keyframe_at_time(project: &mut Project, time: f32) -> Option<usize> {
    if let Some(index) = keyframe_index_at_time(&project.keyframes, time, project.frame_rate) {
        return Some(index);
    }

    if project.keyframes.is_empty() {
        return None;
    }
    let smoothed_keys = smooth_keyframes(&project.keyframes, project.path_smoothing);
    let (x, y) = position_on_path(&smoothed_keys, time);
    add_keyframe(project, time, x, y);
    keyframe_index_at_time(&project.keyframes, time, project.frame_rate)
}

fn preview_tracer_zoom_scale(zoom_scale: f32) -> f32 {
    zoom_scale.max(1.0).powf(-1.15).clamp(0.0, 1.0)
}

fn draw_preview_tracer(
    ui: &egui::Ui,
    painter: &egui::Painter,
    rect: Rect,
    project: &Project,
    time: f32,
    zoom_scale: f32,
    glow_texture: &mut Option<egui::TextureHandle>,
) {
    let zoom_compensation = preview_tracer_zoom_scale(zoom_scale);
    let (width, glow) = scaled_tracer_dimensions(&project.tracer, rect.height());
    let width = width * zoom_compensation;
    let glow = glow * zoom_compensation;
    let points = project_trail_points(project, time);
    let raster_scale = PREVIEW_MAX_DIMENSION as f32 / rect.width().max(rect.height()).max(1.0);
    let raster_width = (rect.width() * raster_scale).round().max(1.0) as u32;
    let raster_height = (rect.height() * raster_scale).round().max(1.0) as u32;
    let raster_points: Vec<(f32, f32)> = points
        .iter()
        .map(|(x, y)| (*x * raster_width as f32, *y * raster_height as f32))
        .collect();
    let glow_layer = render_glow_layer(
        raster_width,
        raster_height,
        &raster_points,
        width * raster_scale,
        glow * raster_scale,
        project.tracer.glow_color,
    );
    if let Some(glow_layer) = glow_layer {
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [raster_width as usize, raster_height as usize],
            glow_layer.as_raw(),
        );
        if let Some(texture) = glow_texture {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            *glow_texture = Some(ui.ctx().load_texture(
                "tracer-glow",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
        if let Some(texture) = glow_texture {
            painter.image(
                texture.id(),
                rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }
    let points: Vec<Pos2> = points
        .iter()
        .map(|(x, y)| {
            Pos2::new(
                rect.left() + x * rect.width(),
                rect.top() + y * rect.height(),
            )
        })
        .collect();
    let color = Color32::from_rgb(
        project.tracer.color[0],
        project.tracer.color[1],
        project.tracer.color[2],
    );
    for segment in points.windows(2) {
        draw_preview_segment(painter, segment[0], segment[1], width, color);
    }
    if let Some(point) = points.last() {
        painter.circle_filled(*point, width, color);
    }
}

fn point_in_polygon(point: Pos2, polygon: &[Pos2]) -> bool {
    let mut inside = false;
    let mut previous = polygon.len().saturating_sub(1);
    for (index, current) in polygon.iter().enumerate() {
        let other = polygon[previous];
        if (current.y > point.y) != (other.y > point.y)
            && point.x
                < (other.x - current.x) * (point.y - current.y) / (other.y - current.y) + current.x
        {
            inside = !inside;
        }
        previous = index;
    }
    inside
}

fn mask_fill_mesh(points: &[(f32, f32)], to_screen: impl Fn((f32, f32)) -> Pos2) -> egui::Mesh {
    let fill = Color32::from_rgba_unmultiplied(255, 214, 90, 128);
    let mut mesh = egui::Mesh::default();
    for point in points {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: to_screen(*point),
            uv: egui::epaint::WHITE_UV,
            color: fill,
        });
    }
    for triangle in triangulate_polygon(points) {
        mesh.indices
            .extend(triangle.iter().map(|&index| index as u32));
    }
    mesh
}

fn distance_to_segment(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let length_squared = ab.length_sq();
    let t = if length_squared <= f32::EPSILON {
        0.0
    } else {
        ((point - a).dot(ab) / length_squared).clamp(0.0, 1.0)
    };
    point.distance(a + ab * t)
}

// Drag corners to reshape, click an edge to add a corner, Shift+click a corner to delete it.
fn edit_selected_mask(
    response: &egui::Response,
    rect: Rect,
    project: &mut Project,
    zoom: &mut CanvasZoom,
    time: f32,
    shift: bool,
    controls_hovered: bool,
) {
    let Some(selected) = zoom.selected_mask else {
        return;
    };
    let frame_rate = project.frame_rate;
    let Some(region) = project.masks.get_mut(selected) else {
        zoom.selected_mask = None;
        return;
    };
    let to_screen = |(x, y): (f32, f32)| {
        Pos2::new(
            rect.left() + x * rect.width(),
            rect.top() + y * rect.height(),
        )
    };
    let to_normalized = |position: Pos2| {
        (
            ((position.x - rect.left()) / rect.width()).clamp(0.0, 1.0),
            ((position.y - rect.top()) / rect.height()).clamp(0.0, 1.0),
        )
    };
    let shape: Vec<Pos2> = region.shape_at(time).into_iter().map(to_screen).collect();
    let nearest_vertex = |position: Pos2| {
        shape
            .iter()
            .enumerate()
            .map(|(index, point)| (index, point.distance(position)))
            .filter(|(_, distance)| *distance <= 14.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
    };
    if response.drag_started_by(egui::PointerButton::Primary) && !controls_hovered {
        let start = response.interact_pointer_pos();
        zoom.dragging_vertex = start.and_then(nearest_vertex);
        zoom.dragging_shape = zoom.dragging_vertex.is_none()
            && start.is_some_and(|position| point_in_polygon(position, &shape));
    }
    if zoom.dragging_shape && response.dragged_by(egui::PointerButton::Primary) {
        let delta = response.drag_delta();
        region.translate(
            (delta.x / rect.width(), delta.y / rect.height()),
            time,
            frame_rate,
        );
        ui_cursor_grab(response);
    }
    if let (Some(vertex), Some(position)) = (zoom.dragging_vertex, response.interact_pointer_pos())
    {
        if response.dragged_by(egui::PointerButton::Primary) {
            region.set_vertex(vertex, to_normalized(position), time, frame_rate);
            ui_cursor_grab(response);
        }
    }
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        zoom.dragging_vertex = None;
        zoom.dragging_shape = false;
    }
    if controls_hovered || response.dragged() || !response.clicked_by(egui::PointerButton::Primary)
    {
        return;
    }
    let Some(position) = response.interact_pointer_pos() else {
        return;
    };
    if let Some(vertex) = nearest_vertex(position) {
        if shift {
            region.remove_vertex(vertex);
        }
        return;
    }
    let edge = (0..shape.len())
        .map(|index| {
            (
                index,
                distance_to_segment(position, shape[index], shape[(index + 1) % shape.len()]),
            )
        })
        .filter(|(_, distance)| *distance <= 10.0)
        .min_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((edge_start, _)) = edge {
        region.insert_vertex(edge_start, to_normalized(position), time, frame_rate);
    }
}

fn ui_cursor_grab(response: &egui::Response) {
    response.ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
}

fn draw_preview_masks(
    painter: &egui::Painter,
    rect: Rect,
    video_texture: egui::TextureId,
    project: &Project,
    zoom: &CanvasZoom,
    time: f32,
) {
    let to_screen = |(x, y): (f32, f32)| {
        Pos2::new(
            rect.left() + x * rect.width(),
            rect.top() + y * rect.height(),
        )
    };
    // Repainting the video over the tracer makes the line pass behind the masked object.
    for (region_index, region) in project.masks.iter().enumerate() {
        let shape = region.shape_at(time);
        let mut mesh = egui::Mesh::with_texture(video_texture);
        for point in &shape {
            mesh.vertices.push(egui::epaint::Vertex {
                pos: to_screen(*point),
                uv: Pos2::new(point.0, point.1),
                color: Color32::WHITE,
            });
        }
        for triangle in triangulate_polygon(&shape) {
            mesh.indices
                .extend(triangle.iter().map(|&index| index as u32));
        }
        painter.add(egui::Shape::mesh(mesh));
        let selected = zoom.selected_mask == Some(region_index);
        if zoom.mask_editing || selected {
            let outline: Vec<Pos2> = shape.iter().map(|point| to_screen(*point)).collect();
            let color = if selected {
                Color32::from_rgb(255, 214, 90)
            } else {
                Color32::from_rgb(120, 200, 255)
            };
            painter.add(egui::Shape::mesh(mask_fill_mesh(&shape, to_screen)));
            painter.add(egui::Shape::closed_line(
                outline.clone(),
                Stroke::new(if selected { 2.0_f32 } else { 1.5_f32 }, color),
            ));
            if selected {
                let on_keyframe = region.keyframe_index_at(time, project.frame_rate).is_some();
                for point in outline {
                    painter.circle_filled(point, 5.5, color);
                    painter.circle_stroke(
                        point,
                        5.5,
                        Stroke::new(
                            1.5_f32,
                            if on_keyframe {
                                Color32::from_rgb(255, 76, 48)
                            } else {
                                Color32::BLACK
                            },
                        ),
                    );
                }
            }
        }
    }
    if zoom.mask_editing && !zoom.mask_draft.is_empty() {
        let outline: Vec<Pos2> = zoom
            .mask_draft
            .iter()
            .map(|point| to_screen(*point))
            .collect();
        if zoom.mask_draft.len() >= 3 {
            painter.add(egui::Shape::mesh(mask_fill_mesh(
                &zoom.mask_draft,
                to_screen,
            )));
        }
        painter.add(egui::Shape::line(
            outline.clone(),
            Stroke::new(2.0_f32, Color32::from_rgb(255, 214, 90)),
        ));
        for (index, point) in outline.iter().enumerate() {
            let radius = if index == 0 { 6.0 } else { 3.5 };
            painter.circle_filled(*point, radius, Color32::from_rgb(255, 214, 90));
        }
    }
}

fn scaled_tracer_dimensions(style: &TracerStyle, canvas_height: f32) -> (f32, f32) {
    let scale = canvas_height / TRACER_REFERENCE_HEIGHT;
    (style.width * scale, style.glow * scale)
}

fn draw_preview_segment(
    painter: &egui::Painter,
    start: Pos2,
    end: Pos2,
    width: f32,
    color: Color32,
) {
    let distance = start.distance(end);
    if distance <= 0.0 {
        painter.line_segment([start, end], Stroke::new(width, color));
        return;
    }
    painter.line_segment([start, end], Stroke::new(width, color));
}

fn probe_video_info(path: &Path) -> Result<(f32, f32), String> {
    Ok((probe_duration(path)?, probe_frame_rate(path)?))
}

// An FFmpeg command that decodes on the GPU when possible. FFmpeg falls back to software
// decoding on its own if the codec or hardware isn't supported.
fn ffmpeg_decoding_command() -> Result<Command, String> {
    let mut command = multimedia_tool_command("ffmpeg")?;
    command.args([
        "-hwaccel",
        if cfg!(target_os = "macos") {
            "videotoolbox"
        } else {
            "auto"
        },
    ]);
    Ok(command)
}

fn multimedia_tool_command(tool: &str) -> Result<Command, String> {
    #[cfg(target_os = "macos")]
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|directory| directory.join(tool)))
        .filter(|candidate| candidate.is_file());
    #[cfg(not(target_os = "macos"))]
    let bundled: Option<PathBuf> = None;

    let from_path = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(multimedia_tool_filename(tool)))
        .find(|candidate| candidate.is_file());
    #[cfg(target_os = "macos")]
    let from_homebrew = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"]
        .into_iter()
        .map(|directory| Path::new(directory).join(tool))
        .find(|candidate| candidate.is_file());
    #[cfg(not(target_os = "macos"))]
    let from_homebrew: Option<PathBuf> = None;

    let executable = bundled.or(from_path).or(from_homebrew).ok_or_else(|| {
        format!("Could not find {tool}. Install FFmpeg and ensure it is available.")
    })?;
    Ok(Command::new(executable))
}

fn multimedia_tool_filename(tool: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        if Path::new(tool).extension().is_none() {
            return format!("{tool}.exe");
        }
    }
    tool.to_owned()
}

fn probe_frame_rate(path: &Path) -> Result<f32, String> {
    let output = multimedia_tool_command("ffprobe")?
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .map_err(|error| {
            format!(
                "Could not start ffprobe. Install FFmpeg, add its bin folder to PATH, and restart Tommy's Tracer: {error}"
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "ffprobe could not read the video's frame rate: {}. Install FFmpeg and ensure ffprobe is available on PATH.",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    for (name, value) in String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_once('='))
    {
        if matches!(name, "avg_frame_rate" | "r_frame_rate") {
            if let Some(frame_rate) = parse_frame_rate(value) {
                return Ok(frame_rate);
            }
        }
    }
    let details = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    Err(format!(
        "Could not determine a valid video frame rate with ffprobe. It reported: {}. Check that the selected file contains a video stream and that FFmpeg is installed.",
        if details.is_empty() {
            "no frame-rate values".to_owned()
        } else {
            details
        }
    ))
}

fn parse_frame_rate(value: &str) -> Option<f32> {
    let value = value.trim();
    let frame_rate = if let Some((numerator, denominator)) = value.split_once('/') {
        let numerator = numerator.parse::<f32>().ok()?;
        let denominator = denominator.parse::<f32>().ok()?;
        numerator / denominator
    } else {
        value.parse::<f32>().ok()?
    };
    (frame_rate.is_finite() && frame_rate > 0.0).then_some(frame_rate)
}

fn probe_duration(path: &Path) -> Result<f32, String> {
    let output = multimedia_tool_command("ffprobe")?
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .map_err(|error| {
            format!("Could not start ffprobe. Install FFmpeg and add it to PATH: {error}")
        })?;
    if !output.status.success() {
        return Err(format!(
            "Could not read video duration: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f32>()
        .map_err(|error| format!("ffprobe returned an invalid duration: {error}"))
}

fn preview_worker(
    commands: Receiver<PreviewCommand>,
    results: mpsc::SyncSender<PreviewResult>,
    repaint_context: egui::Context,
) {
    let mut worker = PreviewWorker::new();
    loop {
        let command = if worker.is_busy() {
            match commands.recv_timeout(Duration::from_millis(2)) {
                Ok(command) => Some(command),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else if worker.is_filling() {
            match commands.recv_timeout(Duration::from_millis(25)) {
                Ok(command) => Some(command),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match commands.recv() {
                Ok(command) => Some(command),
                Err(_) => break,
            }
        };
        if let Some(mut command) = command {
            while let Ok(newer_command) = commands.try_recv() {
                command = newer_command;
            }
            worker.handle_command(command);
        }
        worker.maintain_fill();
        if let Some(message) = worker.step_foreground() {
            match message {
                ForegroundMessage::Reliable(result) => {
                    let _ = results.send(result);
                }
                ForegroundMessage::Droppable(result) => {
                    let _ = results.try_send(result);
                }
            }
            repaint_context.request_repaint();
        }
    }
    worker.stop_filler();
}

enum ForegroundMessage {
    Reliable(PreviewResult),
    Droppable(PreviewResult),
}

enum Foreground {
    Idle,
    Seek {
        id: u64,
        frame: u64,
    },
    Play {
        id: u64,
        frame: u64,
        clock_start: Instant,
        clock_frame: u64,
    },
}

struct PreviewWorker {
    cache: Arc<Mutex<FrameCache>>,
    source: Option<(PathBuf, f32)>,
    frame_range: (u64, u64),
    filled_range: Option<(u64, u64)>,
    filler: Option<FrameFiller>,
    queue: VecDeque<(u64, u64)>,
    last_restart: Option<u64>,
    foreground: Foreground,
    decoded: HashMap<u64, Receiver<Result<PreviewFrame, String>>>,
    buffering: bool,
}

impl PreviewWorker {
    fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(FrameCache::new(PREVIEW_CACHE_BUDGET_BYTES))),
            source: None,
            frame_range: (0, 0),
            filled_range: None,
            filler: None,
            queue: VecDeque::new(),
            last_restart: None,
            foreground: Foreground::Idle,
            decoded: HashMap::new(),
            buffering: false,
        }
    }

    fn is_busy(&self) -> bool {
        !matches!(self.foreground, Foreground::Idle)
    }

    fn is_filling(&self) -> bool {
        self.filler.is_some()
            || !self.queue.is_empty()
            || self.source.is_some() && self.filled_range != Some(self.frame_range)
    }

    fn frame_rate(&self) -> f32 {
        self.source.as_ref().map_or(30.0, |(_, rate)| *rate)
    }

    fn stop_filler(&mut self) {
        if let Some(mut filler) = self.filler.take() {
            filler.stop();
        }
    }

    fn handle_command(&mut self, command: PreviewCommand) {
        self.decoded.clear();
        match command {
            PreviewCommand::Seek(request) => {
                let frame = self.begin_request(&request);
                self.foreground = Foreground::Seek {
                    id: request.id,
                    frame,
                };
            }
            PreviewCommand::Play(request) => {
                let frame = self.begin_request(&request);
                self.buffering = true;
                self.foreground = Foreground::Play {
                    id: request.id,
                    frame,
                    clock_start: Instant::now(),
                    clock_frame: frame,
                };
            }
            PreviewCommand::Pause => self.foreground = Foreground::Idle,
        }
    }

    fn begin_request(&mut self, request: &PreviewRequest) -> u64 {
        let same_source = self.source.as_ref().is_some_and(|(path, rate)| {
            path == &request.path && (rate - request.frame_rate).abs() < 0.001
        });
        if !same_source {
            self.stop_filler();
            self.queue.clear();
            self.filled_range = None;
            self.last_restart = None;
            lock_cache(&self.cache).clear();
            self.source = Some((request.path.clone(), request.frame_rate));
        }
        self.frame_range = request.frame_range;
        let frame = time_to_frame(request.time, request.frame_rate);
        lock_cache(&self.cache).focus = frame;
        frame
    }

    fn maintain_fill(&mut self) {
        if self
            .filler
            .as_ref()
            .is_some_and(|filler| filler.is_finished())
        {
            self.stop_filler();
        }
        if self.filler.is_some() {
            return;
        }
        let Some((path, frame_rate)) = self.source.clone() else {
            return;
        };
        if self.queue.is_empty() && self.filled_range != Some(self.frame_range) {
            let (start, end) = self.frame_range;
            let focus = lock_cache(&self.cache).focus.clamp(start, end);
            self.queue.push_back((focus, end));
            if focus > start {
                self.queue.push_back((start, focus - 1));
            }
            self.filled_range = Some(self.frame_range);
        }
        while let Some((start, end)) = self.queue.pop_front() {
            let Some((start, end)) = lock_cache(&self.cache).uncached_span(start, end) else {
                continue;
            };
            if let Ok(filler) =
                FrameFiller::start(&path, frame_rate, start, end, Arc::clone(&self.cache))
            {
                self.filler = Some(filler);
                break;
            }
        }
    }

    // Makes sure a decoder is producing `frame` soon. Err means it can never be produced.
    fn ensure_frame_fill(&mut self, frame: u64) -> Result<(), String> {
        let active = self
            .filler
            .as_ref()
            .is_some_and(|filler| !filler.is_finished());
        if let Some(filler) = self.filler.as_ref().filter(|_| active) {
            let progress = filler.progress();
            if frame >= progress
                && frame <= filler.end
                && frame - progress <= PREVIEW_FILL_LOOKAHEAD_FRAMES
            {
                return Ok(());
            }
        }
        if self.last_restart == Some(frame) {
            return if active || !self.queue.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "Frame {frame} could not be decoded from the video."
                ))
            };
        }
        self.last_restart = Some(frame);
        self.stop_filler();
        self.queue.clear();
        let (start, end) = self.frame_range;
        let (start, end) = (start.min(frame), end.max(frame));
        self.queue.push_back((frame, end));
        if frame > start {
            self.queue.push_back((start, frame - 1));
        }
        self.filled_range = Some(self.frame_range);
        self.maintain_fill();
        if self.filler.is_none() {
            return Err("Could not start FFmpeg preview decoding.".to_owned());
        }
        Ok(())
    }

    fn cached_frame(&self, frame: u64) -> Option<Arc<Vec<u8>>> {
        lock_cache(&self.cache).get(frame)
    }

    // JPEG decoding is slower than a 60 fps frame period, so upcoming frames decode in parallel.
    fn start_decode_ahead(&mut self, from: u64) {
        self.decoded.retain(|frame, _| *frame >= from);
        let last = self
            .frame_range
            .1
            .min(from + PREVIEW_DECODE_AHEAD_FRAMES - 1);
        for frame in from..=last {
            if self.decoded.contains_key(&frame) {
                continue;
            }
            let Some(bytes) = self.cached_frame(frame) else {
                break;
            };
            let (sender, receiver) = mpsc::channel();
            std::thread::spawn(move || {
                let _ = sender.send(decode_preview_frame(&bytes));
            });
            self.decoded.insert(frame, receiver);
        }
    }

    fn decoded_frame(&mut self, frame: u64, bytes: &[u8]) -> Result<PreviewFrame, String> {
        match self.decoded.remove(&frame) {
            Some(receiver) => receiver
                .recv()
                .unwrap_or_else(|_| decode_preview_frame(bytes)),
            None => decode_preview_frame(bytes),
        }
    }

    fn step_foreground(&mut self) -> Option<ForegroundMessage> {
        let frame_rate = self.frame_rate();
        match self.foreground {
            Foreground::Idle => None,
            Foreground::Seek { id, frame } => {
                if let Some(bytes) = self.cached_frame(frame) {
                    self.foreground = Foreground::Idle;
                    return Some(ForegroundMessage::Reliable(PreviewResult {
                        id,
                        time: frame_time(frame, frame_rate),
                        result: decode_preview_frame(&bytes),
                        finished: false,
                    }));
                }
                match self.ensure_frame_fill(frame) {
                    Ok(()) => None,
                    Err(error) => {
                        self.foreground = Foreground::Idle;
                        Some(ForegroundMessage::Reliable(PreviewResult {
                            id,
                            time: frame_time(frame, frame_rate),
                            result: Err(error),
                            finished: false,
                        }))
                    }
                }
            }
            Foreground::Play {
                id,
                frame,
                clock_start,
                clock_frame,
            } => {
                if frame > self.frame_range.1 {
                    self.foreground = Foreground::Idle;
                    return Some(ForegroundMessage::Reliable(PreviewResult {
                        id,
                        time: frame_time(frame, frame_rate),
                        result: Ok(PreviewFrame::default()),
                        finished: true,
                    }));
                }
                lock_cache(&self.cache).focus = frame;
                self.start_decode_ahead(frame);
                if self.buffering {
                    let needed = ((frame_rate * PLAYBACK_BUFFER_SECONDS).ceil() as u64)
                        .min(self.frame_range.1 - frame + 1);
                    if lock_cache(&self.cache).cached_run(frame, needed) < needed {
                        return match self.ensure_frame_fill(frame) {
                            Ok(()) => None,
                            Err(_) => {
                                self.foreground = Foreground::Idle;
                                Some(ForegroundMessage::Reliable(PreviewResult {
                                    id,
                                    time: frame_time(frame, frame_rate),
                                    result: Ok(PreviewFrame::default()),
                                    finished: true,
                                }))
                            }
                        };
                    }
                    self.buffering = false;
                    self.foreground = Foreground::Play {
                        id,
                        frame,
                        clock_start: Instant::now(),
                        clock_frame: frame,
                    };
                    return None;
                }
                let deadline =
                    playback_frame_deadline(clock_start, frame - clock_frame, frame_rate);
                if Instant::now() < deadline {
                    return None;
                }
                if let Some(bytes) = self.cached_frame(frame) {
                    self.foreground = Foreground::Play {
                        id,
                        frame: frame + 1,
                        clock_start,
                        clock_frame,
                    };
                    return Some(ForegroundMessage::Droppable(PreviewResult {
                        id,
                        time: frame_time(frame, frame_rate),
                        result: self.decoded_frame(frame, &bytes),
                        finished: false,
                    }));
                }
                // Decoding is behind playback: pause and rebuffer instead of stuttering.
                self.buffering = true;
                match self.ensure_frame_fill(frame) {
                    Ok(()) => None,
                    Err(_) => {
                        self.foreground = Foreground::Idle;
                        Some(ForegroundMessage::Reliable(PreviewResult {
                            id,
                            time: frame_time(frame, frame_rate),
                            result: Ok(PreviewFrame::default()),
                            finished: true,
                        }))
                    }
                }
            }
        }
    }
}

fn lock_cache(cache: &Mutex<FrameCache>) -> std::sync::MutexGuard<'_, FrameCache> {
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Compressed preview frames kept in memory; frames far from the playhead are evicted first.
struct FrameCache {
    frames: BTreeMap<u64, Arc<Vec<u8>>>,
    bytes: usize,
    budget: usize,
    focus: u64,
}

impl FrameCache {
    fn new(budget: usize) -> Self {
        Self {
            frames: BTreeMap::new(),
            bytes: 0,
            budget,
            focus: 0,
        }
    }

    fn clear(&mut self) {
        self.frames.clear();
        self.bytes = 0;
    }

    fn get(&self, frame: u64) -> Option<Arc<Vec<u8>>> {
        self.frames.get(&frame).cloned()
    }

    // Returns false when the frame is farther from the playhead than everything already cached.
    fn insert(&mut self, frame: u64, data: Vec<u8>) -> bool {
        if let Some(old) = self.frames.remove(&frame) {
            self.bytes -= old.len();
        }
        let size = data.len();
        while self.bytes + size > self.budget {
            let first = self.frames.keys().next().copied();
            let last = self.frames.keys().next_back().copied();
            let (Some(first), Some(last)) = (first, last) else {
                return false;
            };
            let farthest = if self.focus.abs_diff(first) >= self.focus.abs_diff(last) {
                first
            } else {
                last
            };
            if self.focus.abs_diff(farthest) <= self.focus.abs_diff(frame) {
                return false;
            }
            if let Some(removed) = self.frames.remove(&farthest) {
                self.bytes -= removed.len();
            }
        }
        self.bytes += size;
        self.frames.insert(frame, Arc::new(data));
        true
    }

    fn cached_run(&self, start: u64, limit: u64) -> u64 {
        (0..limit)
            .take_while(|offset| self.frames.contains_key(&(start + offset)))
            .count() as u64
    }

    // Shrinks start..=end past already cached frames at both ends.
    fn uncached_span(&self, mut start: u64, mut end: u64) -> Option<(u64, u64)> {
        while start <= end && self.frames.contains_key(&start) {
            start += 1;
        }
        while end >= start && self.frames.contains_key(&end) {
            end -= 1;
        }
        (start <= end).then_some((start, end))
    }
}

// Background FFmpeg decode of a frame range into the cache.
struct FrameFiller {
    child: Child,
    end: u64,
    progress: Arc<AtomicU64>,
    finished: Arc<AtomicBool>,
}

impl FrameFiller {
    fn start(
        path: &Path,
        frame_rate: f32,
        start: u64,
        end: u64,
        cache: Arc<Mutex<FrameCache>>,
    ) -> Result<Self, String> {
        let dimension = PREVIEW_FRAME_MAX_DIMENSION;
        let mut command = ffmpeg_decoding_command()?;
        command.args(["-v", "error"]);
        let mut child = command
            .arg("-ss")
            .arg(format!("{:.6}", frame_time(start, frame_rate)))
            .arg("-i")
            .arg(path)
            .args(["-an", "-sn", "-dn", "-vf"])
            .arg(format!(
                "fps={frame_rate:.6},scale='if(gt(iw,ih),min({dimension},iw),-2)':'if(gt(iw,ih),-2,min({dimension},ih))'"
            ))
            .arg("-frames:v")
            .arg((end - start + 1).to_string())
            .args(["-q:v", "7", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("Could not start FFmpeg preview decoding: {error}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Could not read FFmpeg preview frames.".to_owned())?;
        let progress = Arc::new(AtomicU64::new(start));
        let finished = Arc::new(AtomicBool::new(false));
        let thread_progress = Arc::clone(&progress);
        let thread_finished = Arc::clone(&finished);
        let spawned = std::thread::Builder::new()
            .name("opentracer-frame-filler".to_owned())
            .spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut buffered = Vec::new();
                let mut frame = start;
                while frame <= end {
                    let Ok(Some(bytes)) = read_mjpeg_frame(&mut reader, &mut buffered) else {
                        break;
                    };
                    let accepted = lock_cache(&cache).insert(frame, bytes);
                    frame += 1;
                    thread_progress.store(frame, Ordering::Release);
                    if !accepted {
                        break;
                    }
                }
                thread_finished.store(true, Ordering::Release);
            });
        if let Err(error) = spawned {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("Could not start video frame reader: {error}"));
        }
        Ok(Self {
            child,
            end,
            progress,
            finished,
        })
    }

    fn progress(&self) -> u64 {
        self.progress.load(Ordering::Acquire)
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn decode_preview_frame(bytes: &[u8]) -> Result<PreviewFrame, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| format!("Could not decode video preview: {error}"))?
        .to_rgba8();
    Ok(PreviewFrame {
        size: [image.width() as usize, image.height() as usize],
        rgba: image.into_raw(),
    })
}

fn analyze_camera_motion(
    project: &Project,
    cancelled: &AtomicBool,
    mut report_progress: impl FnMut(usize, usize),
) -> Result<Vec<CameraMotionSample>, String> {
    const ANALYSIS_MAX_DIMENSION: u32 = 480;
    let width = probe_video_dimension(&project.video_path, "width")?;
    let height = probe_video_dimension(&project.video_path, "height")?;
    if width == 0 || height == 0 || project.trim_out <= project.trim_in {
        return Err("The selected video trim cannot be analyzed.".to_owned());
    }
    let scale = (ANALYSIS_MAX_DIMENSION as f32 / width.max(height) as f32).min(1.0);
    let analysis_width = (width as f32 * scale).round().max(1.0) as u32;
    let analysis_height = (height as f32 * scale).round().max(1.0) as u32;
    let frame_size = (analysis_width * analysis_height) as usize;
    let duration = project.trim_out - project.trim_in;
    let analysis_fps = motion_analysis_fps(project.frame_rate);
    let expected_frames = expected_motion_analysis_samples(duration, analysis_fps);

    let mut child = ffmpeg_decoding_command()?
        .args(["-v", "error", "-ss"])
        .arg(format!("{:.6}", project.trim_in))
        .arg("-i")
        .arg(&project.video_path)
        .arg("-t")
        .arg(format!("{duration:.6}"))
        .arg("-vf")
        .arg(format!(
            "fps={analysis_fps:.3},scale={analysis_width}:{analysis_height}:flags=area,format=gray"
        ))
        .args(["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not start FFmpeg camera-motion analysis: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Could not read FFmpeg motion-analysis frames.".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Could not read FFmpeg motion-analysis errors.".to_owned())?;
    let stderr_reader = std::thread::Builder::new()
        .name("opentracer-motion-errors".to_owned())
        .spawn(move || {
            let mut output = Vec::new();
            BufReader::new(stderr)
                .read_to_end(&mut output)
                .map(|_| output)
        })
        .map_err(|error| {
            let _ = child.kill();
            let _ = child.wait();
            format!("Could not start FFmpeg error reader: {error}")
        })?;
    let mut reader = BufReader::new(stdout);
    let mut samples = Vec::with_capacity(expected_frames);
    let mut previous_frame: Option<Vec<u8>> = None;
    let mut cumulative_x = 0.0_f32;
    let mut cumulative_y = 0.0_f32;
    let mut last_delta = (0.0_f32, 0.0_f32);
    loop {
        if cancelled.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stderr_reader.join();
            return Err("Camera-motion analysis was cancelled.".to_owned());
        }
        let frame = match read_raw_frame(&mut reader, frame_size) {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stderr_reader.join();
                return Err(format!("Could not read motion-analysis frame: {error}"));
            }
        };

        if let Some(previous) = &previous_frame {
            // When a frame can't be matched (heavy blur), keep drifting along the recent motion.
            let (delta_x, delta_y) =
                estimate_frame_translation(previous, &frame, analysis_width, analysis_height)
                    .unwrap_or((last_delta.0 * 0.5, last_delta.1 * 0.5));
            last_delta = (delta_x, delta_y);
            cumulative_x =
                (cumulative_x + delta_x).clamp(-(analysis_width as f32), analysis_width as f32);
            cumulative_y =
                (cumulative_y + delta_y).clamp(-(analysis_height as f32), analysis_height as f32);
        }
        let sample_index = samples.len();
        samples.push(CameraMotionSample {
            time: project.trim_in + sample_index as f32 / analysis_fps,
            x: cumulative_x / analysis_width as f32,
            y: cumulative_y / analysis_height as f32,
        });
        previous_frame = Some(frame);
        report_progress(samples.len().min(expected_frames), expected_frames);
    }

    let status = child
        .wait()
        .map_err(|error| format!("Could not finish FFmpeg motion analysis: {error}"))?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "FFmpeg error reader stopped unexpectedly.".to_owned())?
        .map_err(|error| format!("Could not read FFmpeg camera-motion errors: {error}"))?;
    if !status.success() {
        return Err(format!(
            "FFmpeg camera-motion analysis failed: {}",
            String::from_utf8_lossy(&stderr).trim()
        ));
    }
    if samples.is_empty() {
        return Err("FFmpeg returned no frames for motion analysis.".to_owned());
    }
    report_progress(samples.len(), samples.len());
    Ok(samples)
}

const BALL_TRACK_MAX_DIMENSION: u32 = 1280;
const BALL_RING_RADIUS: i32 = 6;
const BALL_MIN_CONTRAST: f32 = 10.0;

fn lagrange_quadratic(points: [(f32, f32, f32); 3], time: f32) -> (f32, f32) {
    let mut x = 0.0;
    let mut y = 0.0;
    for i in 0..3 {
        let mut weight = 1.0;
        for j in 0..3 {
            if i != j {
                weight *= (time - points[j].0) / (points[i].0 - points[j].0);
            }
        }
        x += weight * points[i].1;
        y += weight * points[i].2;
    }
    (x, y)
}

// Smooth ballistic-style arc through the user's anchors `(time, x, y)`: every segment blends
// the parabolas through its neighbouring triples, so the arc passes exactly through each anchor.
fn predict_ball_position(anchors: &[(f32, f32, f32)], time: f32) -> (f32, f32) {
    let Some(first) = anchors.first() else {
        return (0.0, 0.0);
    };
    let last = anchors[anchors.len() - 1];
    if time <= first.0 || anchors.len() == 1 {
        return (first.1, first.2);
    }
    if time >= last.0 {
        return (last.1, last.2);
    }
    let segment = anchors.partition_point(|anchor| anchor.0 <= time) - 1;
    let (a, b) = (anchors[segment], anchors[segment + 1]);
    let weight = ((time - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
    let linear = (a.1 + (b.1 - a.1) * weight, a.2 + (b.2 - a.2) * weight);
    let before = (segment > 0).then(|| lagrange_quadratic([anchors[segment - 1], a, b], time));
    let after = (segment + 2 < anchors.len())
        .then(|| lagrange_quadratic([a, b, anchors[segment + 2]], time));
    match (before, after) {
        (Some(before), Some(after)) => (
            before.0 * (1.0 - weight) + after.0 * weight,
            before.1 * (1.0 - weight) + after.1 * weight,
        ),
        (Some(only), None) | (None, Some(only)) => only,
        (None, None) => linear,
    }
}

fn core_brightness(frame: &[u8], width: usize, x: usize, y: usize) -> f32 {
    let mut sum = 0_u32;
    for row in y - 1..=y + 1 {
        for column in x - 1..=x + 1 {
            sum += frame[row * width + column] as u32;
        }
    }
    sum as f32 / 9.0
}

// Finds the most ball-like small blob near `center`: a spot that differs from its
// surroundings and, when neighbouring frames are available, from the same spot a frame
// before/after (so static bright things are ignored). Returns (x, y, contrast).
fn find_ball(
    previous: Option<&[u8]>,
    current: &[u8],
    next: Option<&[u8]>,
    width: usize,
    height: usize,
    center: (f32, f32),
    radius: f32,
) -> Option<(f32, f32, f32)> {
    let margin = BALL_RING_RADIUS as usize + 2;
    if width <= margin * 2 || height <= margin * 2 {
        return None;
    }
    let ring = BALL_RING_RADIUS;
    let diagonal = (BALL_RING_RADIUS as f32 * 0.707).round() as i32;
    let ring_offsets = [
        (ring, 0),
        (-ring, 0),
        (0, ring),
        (0, -ring),
        (diagonal, diagonal),
        (-diagonal, diagonal),
        (diagonal, -diagonal),
        (-diagonal, -diagonal),
    ];
    let x_min = ((center.0 - radius).floor().max(margin as f32)) as usize;
    let x_max = ((center.0 + radius).ceil().min((width - margin - 1) as f32)) as usize;
    let y_min = ((center.1 - radius).floor().max(margin as f32)) as usize;
    let y_max = ((center.1 + radius).ceil().min((height - margin - 1) as f32)) as usize;
    let sigma = (radius * 0.6).max(1.0);
    let mut best: Option<(f32, f32, f32, f32)> = None;
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let dx = x as f32 - center.0;
            let dy = y as f32 - center.1;
            let distance_sq = dx * dx + dy * dy;
            if distance_sq > radius * radius {
                continue;
            }
            let core = core_brightness(current, width, x, y);
            let mut ring_max = f32::MIN;
            let mut ring_min = f32::MAX;
            for (ox, oy) in ring_offsets {
                let value =
                    current[((y as i32 + oy) as usize) * width + (x as i32 + ox) as usize] as f32;
                ring_max = ring_max.max(value);
                ring_min = ring_min.min(value);
            }
            let contrast = (core - ring_max).max((ring_min - core) * 0.7);
            if contrast < BALL_MIN_CONTRAST {
                continue;
            }
            let mut motion: Option<f32> = None;
            for neighbour in [previous, next].into_iter().flatten() {
                let difference = (core - core_brightness(neighbour, width, x, y)).abs();
                motion = Some(motion.map_or(difference, |value| value.min(difference)));
            }
            let score = contrast
                * (1.0 + motion.unwrap_or(0.0) / 32.0)
                * (-distance_sq / (2.0 * sigma * sigma)).exp();
            if best.is_none_or(|(_, _, _, best_score)| score > best_score) {
                best = Some((x as f32, y as f32, contrast, score));
            }
        }
    }
    best.map(|(x, y, contrast, _)| (x, y, contrast))
}

const BALL_COLOUR_RING: i32 = 9;
const BALL_COLOUR_TOLERANCE: f32 = 60.0;
const BALL_MIN_SATURATION: f32 = 40.0;

struct TrackFrame {
    gray: Vec<u8>,
    rgb: Vec<u8>,
}

impl TrackFrame {
    fn new(rgb: Vec<u8>) -> Self {
        let gray = rgb
            .chunks(3)
            .map(|p| ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8)
            .collect();
        Self { gray, rgb }
    }

    // Colour difference from brightness (Cb, Cr), so shadows do not change a ball's colour much.
    fn chroma(&self, index: usize) -> (f32, f32) {
        let p = &self.rgb[index * 3..index * 3 + 3];
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        (b - y, r - y)
    }

    fn colour_match(&self, index: usize, reference: (f32, f32)) -> f32 {
        let (cb, cr) = self.chroma(index);
        let distance = ((cb - reference.0).powi(2) + (cr - reference.1).powi(2)).sqrt();
        (1.0 - distance / BALL_COLOUR_TOLERANCE).max(0.0)
    }
}

fn rgb_chroma(rgb: [u8; 3]) -> (f32, f32) {
    let (r, g, b) = (rgb[0] as f32, rgb[1] as f32, rgb[2] as f32);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    (b - y, r - y)
}

// Picks the most saturated 3x3 patch within a few pixels of a click on the preview
// (normalized position) and returns its average colour, or None if it is white/grey.
fn pick_ball_colour(rgba: &[u8], size: [usize; 2], point: (f32, f32)) -> Option<[u8; 3]> {
    let (width, height) = (size[0], size[1]);
    if width < 5 || height < 5 || rgba.len() < width * height * 4 {
        return None;
    }
    let cx = (point.0 * width as f32).round() as i32;
    let cy = (point.1 * height as f32).round() as i32;
    let mut best: Option<(f32, [u8; 3])> = None;
    for y in (cy - 4).max(1)..=(cy + 4).min(height as i32 - 2) {
        for x in (cx - 4).max(1)..=(cx + 4).min(width as i32 - 2) {
            let mut sum = [0_u32; 3];
            for row in y - 1..=y + 1 {
                for column in x - 1..=x + 1 {
                    let index = (row as usize * width + column as usize) * 4;
                    for channel in 0..3 {
                        sum[channel] += rgba[index + channel] as u32;
                    }
                }
            }
            let colour = [(sum[0] / 9) as u8, (sum[1] / 9) as u8, (sum[2] / 9) as u8];
            let (cb, cr) = rgb_chroma(colour);
            let saturation = (cb * cb + cr * cr).sqrt();
            if best.is_none_or(|(value, _)| saturation > value) {
                best = Some((saturation, colour));
            }
        }
    }
    best.filter(|(saturation, _)| *saturation >= BALL_MIN_SATURATION)
        .map(|(_, colour)| colour)
}

// Learns the ball's colour from the most saturated 3x3 patch around a clicked position.
fn sample_ball_colour(
    frame: &TrackFrame,
    width: usize,
    height: usize,
    center: (f32, f32),
) -> Option<(f32, f32)> {
    let margin = 2_usize;
    let mut best: Option<(f32, (f32, f32))> = None;
    let (cx, cy) = (center.0.round() as i32, center.1.round() as i32);
    for y in cy - 12..=cy + 12 {
        for x in cx - 12..=cx + 12 {
            if x < margin as i32
                || y < margin as i32
                || x >= (width - margin) as i32
                || y >= (height - margin) as i32
            {
                continue;
            }
            let (mut cb, mut cr) = (0.0, 0.0);
            for row in y - 1..=y + 1 {
                for column in x - 1..=x + 1 {
                    let (b, r) = frame.chroma(row as usize * width + column as usize);
                    cb += b / 9.0;
                    cr += r / 9.0;
                }
            }
            let saturation = (cb * cb + cr * cr).sqrt();
            if best.is_none_or(|(value, _)| saturation > value) {
                best = Some((saturation, (cb, cr)));
            }
        }
    }
    best.filter(|(saturation, _)| *saturation >= BALL_MIN_SATURATION)
        .map(|(_, colour)| colour)
}

// Finds the blob that best matches the ball's colour and stands out from its surroundings.
fn find_coloured_ball(
    frame: &TrackFrame,
    width: usize,
    height: usize,
    center: (f32, f32),
    radius: f32,
    reference: (f32, f32),
) -> Option<(f32, f32, f32)> {
    let margin = BALL_COLOUR_RING as usize + 2;
    if width <= margin * 2 || height <= margin * 2 {
        return None;
    }
    let ring = BALL_COLOUR_RING;
    let diagonal = (ring as f32 * 0.707).round() as i32;
    let ring_offsets = [
        (ring, 0),
        (-ring, 0),
        (0, ring),
        (0, -ring),
        (diagonal, diagonal),
        (-diagonal, diagonal),
        (diagonal, -diagonal),
        (-diagonal, -diagonal),
    ];
    let x_min = (center.0 - radius).floor().max(margin as f32) as usize;
    let x_max = (center.0 + radius).ceil().min((width - margin - 1) as f32) as usize;
    let y_min = (center.1 - radius).floor().max(margin as f32) as usize;
    let y_max = (center.1 + radius).ceil().min((height - margin - 1) as f32) as usize;
    let sigma = (radius * 0.8).max(1.0);
    let mut best: Option<(f32, f32, f32, f32)> = None;
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let dx = x as f32 - center.0;
            let dy = y as f32 - center.1;
            let distance_sq = dx * dx + dy * dy;
            if distance_sq > radius * radius {
                continue;
            }
            if frame.colour_match(y * width + x, reference) < 0.15 {
                continue;
            }
            let mut core = 0.0;
            for row in y - 1..=y + 1 {
                for column in x - 1..=x + 1 {
                    core += frame.colour_match(row * width + column, reference) / 9.0;
                }
            }
            if core < 0.35 {
                continue;
            }
            let mut ring_max = 0.0_f32;
            for (ox, oy) in ring_offsets {
                let index = ((y as i32 + oy) as usize) * width + (x as i32 + ox) as usize;
                ring_max = ring_max.max(frame.colour_match(index, reference));
            }
            let contrast = core - ring_max;
            if contrast < 0.2 {
                continue;
            }
            let score = contrast * (-distance_sq / (2.0 * sigma * sigma)).exp();
            if best.is_none_or(|(_, _, _, best_score)| score > best_score) {
                best = Some((x as f32, y as f32, contrast, score));
            }
        }
    }
    best.map(|(x, y, contrast, _)| (x, y, contrast))
}

#[derive(Clone, Copy)]
struct TrackSettings {
    width: usize,
    height: usize,
    frame_rate: f32,
    picked_colour: Option<(f32, f32)>,
}

// Follows the ball through the clip between the first and last tracer keyframes.
// Frames are RGB24. Returns the frame-by-frame positions in image pixels, `(time, x, y)`.
fn follow_ball(
    frames: impl FnMut() -> Result<Option<Vec<u8>>, String>,
    anchors: &[(f32, f32, f32)],
    settings: TrackSettings,
    mut report_progress: impl FnMut(usize, usize),
    cancelled: &AtomicBool,
) -> Result<Vec<(f32, f32, f32)>, String> {
    let TrackSettings {
        width,
        height,
        frame_rate,
        picked_colour,
    } = settings;
    let mut read_frame = frames;
    let start = anchors[0].0;
    let end = anchors[anchors.len() - 1].0;
    let total = (((end - start) * frame_rate).round() as usize).max(1) + 1;
    let max_dimension = width.max(height) as f32;
    let max_radius = 0.06 * max_dimension;
    let anchor_frames: Vec<usize> = anchors
        .iter()
        .map(|anchor| ((anchor.0 - start) * frame_rate).round() as usize)
        .collect();
    let mut positions = Vec::with_capacity(total);
    let mut offset = (0.0_f32, 0.0_f32);
    let mut colour = picked_colour;
    let mut previous: Option<TrackFrame> = None;
    let Some(first) = read_frame()? else {
        return Err("FFmpeg returned no frames for ball tracking.".to_owned());
    };
    let mut current = TrackFrame::new(first);
    for index in 0..total {
        if cancelled.load(Ordering::Relaxed) {
            return Err("Ball tracing was cancelled.".to_owned());
        }
        let next = match index + 1 < total {
            true => read_frame()?.map(TrackFrame::new),
            false => None,
        };
        let time = start + index as f32 / frame_rate;
        for (anchor, anchor_frame) in anchors.iter().zip(&anchor_frames) {
            if *anchor_frame == index && picked_colour.is_none() {
                if let Some(sampled) =
                    sample_ball_colour(&current, width, height, (anchor.1, anchor.2))
                {
                    colour = Some(match colour {
                        Some(known) => ((known.0 + sampled.0) * 0.5, (known.1 + sampled.1) * 0.5),
                        None => sampled,
                    });
                }
            }
        }
        let predicted = predict_ball_position(anchors, time);
        let nearest_anchor = anchors
            .iter()
            .map(|anchor| (anchor.0 - time).abs())
            .fold(f32::MAX, f32::min);
        let closeness = (nearest_anchor / 0.25).clamp(0.0, 1.0);
        let follow = (nearest_anchor / 0.15).clamp(0.0, 1.0);
        let center = (
            predicted.0 + offset.0 * follow,
            predicted.1 + offset.1 * follow,
        );
        // A known ball colour allows a much wider search, so a rough arc is good enough.
        let found = colour
            .and_then(|reference| {
                let radius = 24.0 + 0.2 * max_dimension * closeness.max(0.35);
                find_coloured_ball(&current, width, height, center, radius, reference)
            })
            .or_else(|| {
                find_ball(
                    previous.as_ref().map(|frame| frame.gray.as_slice()),
                    &current.gray,
                    next.as_ref().map(|frame| frame.gray.as_slice()),
                    width,
                    height,
                    center,
                    8.0 + max_radius * closeness,
                )
            });
        let position = match found {
            Some((x, y, _)) => {
                offset = (x - predicted.0, y - predicted.1);
                (x, y)
            }
            None => center,
        };
        positions.push((time, position.0, position.1));
        report_progress(index + 1, total);
        previous = Some(current);
        match next {
            Some(frame) => current = frame,
            None => break,
        }
    }
    Ok(positions)
}

fn trace_ball(
    project: &Project,
    cancelled: &AtomicBool,
    mut report_progress: impl FnMut(usize, usize),
) -> Result<Vec<Keyframe>, String> {
    let mut keyframes = project.keyframes.clone();
    keyframes.sort_by(|a, b| a.time.total_cmp(&b.time));
    if keyframes.len() < 2 || keyframes[keyframes.len() - 1].time - keyframes[0].time < 0.1 {
        return Err("Add at least two tracer keyframes (ball start and end) first.".to_owned());
    }
    let width = probe_video_dimension(&project.video_path, "width")?;
    let height = probe_video_dimension(&project.video_path, "height")?;
    if width == 0 || height == 0 {
        return Err("Could not read the video size.".to_owned());
    }
    let scale = (BALL_TRACK_MAX_DIMENSION as f32 / width.max(height) as f32).min(1.0);
    let analysis_width = (width as f32 * scale).round().max(1.0) as u32;
    let analysis_height = (height as f32 * scale).round().max(1.0) as u32;
    let frame_size = (analysis_width * analysis_height * 3) as usize;
    let anchors: Vec<(f32, f32, f32)> = keyframes
        .iter()
        .map(|keyframe| {
            let (offset_x, offset_y) = camera_motion_offset(project, keyframe.time);
            (
                keyframe.time,
                (keyframe.x + offset_x) * analysis_width as f32,
                (keyframe.y + offset_y) * analysis_height as f32,
            )
        })
        .collect();
    let start = anchors[0].0;
    let duration = anchors[anchors.len() - 1].0 - start + 1.0 / project.frame_rate;

    let mut child = ffmpeg_decoding_command()?
        .args(["-v", "error", "-ss"])
        .arg(format!("{start:.6}"))
        .arg("-i")
        .arg(&project.video_path)
        .arg("-t")
        .arg(format!("{duration:.6}"))
        .arg("-vf")
        .arg(format!(
            "fps={},scale={analysis_width}:{analysis_height}:flags=area,format=rgb24",
            project.frame_rate
        ))
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Could not start FFmpeg ball tracing: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Could not read FFmpeg ball-tracing frames.".to_owned())?;
    let mut reader = BufReader::new(stdout);
    let result = follow_ball(
        || read_raw_frame(&mut reader, frame_size).map_err(|error| error.to_string()),
        &anchors,
        TrackSettings {
            width: analysis_width as usize,
            height: analysis_height as usize,
            frame_rate: project.frame_rate,
            picked_colour: project.ball_colour.map(rgb_chroma),
        },
        &mut report_progress,
        cancelled,
    );
    let _ = child.kill();
    let _ = child.wait();
    let positions = result?;

    // Roughly ten keyframes per second keep the path editable; the user's own keyframes stay.
    let step = ((project.frame_rate / 10.0).round() as usize).max(1);
    let keep_clear = step as f32 / project.frame_rate * 0.5;
    let mut traced = keyframes.clone();
    for (index, (time, x, y)) in positions.iter().enumerate() {
        if index % step != 0 {
            continue;
        }
        if keyframes
            .iter()
            .any(|keyframe| (keyframe.time - time).abs() < keep_clear.max(0.5 / project.frame_rate))
        {
            continue;
        }
        let (offset_x, offset_y) = camera_motion_offset(project, *time);
        traced.push(Keyframe {
            time: *time,
            x: (x / analysis_width as f32 - offset_x).clamp(0.0, 1.0),
            y: (y / analysis_height as f32 - offset_y).clamp(0.0, 1.0),
        });
    }
    traced.sort_by(|a, b| a.time.total_cmp(&b.time));
    Ok(traced)
}

fn read_raw_frame(reader: &mut impl Read, frame_size: usize) -> std::io::Result<Option<Vec<u8>>> {
    let mut frame = vec![0_u8; frame_size];
    let first_read = reader.read(&mut frame[..1])?;
    if first_read == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut frame[1..])?;
    Ok(Some(frame))
}

fn downsample_half(source: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let (out_width, out_height) = ((width / 2).max(1), (height / 2).max(1));
    let mut output = vec![0_u8; out_width * out_height];
    for y in 0..out_height {
        for x in 0..out_width {
            let (x0, y0) = ((x * 2).min(width - 1), (y * 2).min(height - 1));
            let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
            let sum = source[y0 * width + x0] as u32
                + source[y0 * width + x1] as u32
                + source[y1 * width + x0] as u32
                + source[y1 * width + x1] as u32;
            output[y * out_width + x] = ((sum + 2) / 4) as u8;
        }
    }
    (output, out_width, out_height)
}

// Mean truncated pixel difference between `previous` and `current` shifted by (dx, dy),
// measured over the area where both images overlap.
fn shift_error(
    previous: &[u8],
    current: &[u8],
    width: usize,
    height: usize,
    (dx, dy): (i32, i32),
    step: usize,
) -> Option<f32> {
    const MAX_PIXEL_DIFFERENCE: u8 = 48;
    let x_start = (-dx).max(0) as usize;
    let x_end = (width as i32 - dx.max(0)).max(0) as usize;
    let y_start = (-dy).max(0) as usize;
    let y_end = (height as i32 - dy.max(0)).max(0) as usize;
    if x_end <= x_start + width / 2 || y_end <= y_start + height / 2 {
        return None;
    }
    let mut error = 0_u64;
    let mut count = 0_u64;
    for y in (y_start..y_end).step_by(step) {
        let current_row = (y as i32 + dy) as usize * width;
        for x in (x_start..x_end).step_by(step) {
            let current_index = current_row + (x as i32 + dx) as usize;
            error += previous[y * width + x]
                .abs_diff(current[current_index])
                .min(MAX_PIXEL_DIFFERENCE) as u64;
            count += 1;
        }
    }
    (count > 0).then(|| error as f32 / count as f32)
}

// Offset of the parabola's minimum through three equally spaced samples, within half a step.
fn parabola_offset(before: f32, centre: f32, after: f32) -> f32 {
    let curvature = before - 2.0 * centre + after;
    if curvature <= f32::EPSILON {
        0.0
    } else {
        (0.5 * (before - after) / curvature).clamp(-0.5, 0.5)
    }
}

// Coarse-to-fine block matching: a wide search on a quarter-size image catches fast camera
// moves, then two refinement passes and a sub-pixel fit recover precision.
fn estimate_frame_translation(
    previous: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
) -> Option<(f32, f32)> {
    const COARSE_RADIUS: i32 = 10;
    const REFINE_RADIUS: i32 = 2;
    const MIN_COARSE_CONTRAST: f32 = 0.75;
    const MAX_ACCEPTED_ERROR: f32 = 28.0;
    let (width, height) = (width as usize, height as usize);
    if width < 64 || height < 64 {
        return None;
    }

    let mut pyramid = vec![(previous.to_vec(), current.to_vec(), width, height)];
    for _ in 0..2 {
        let (previous, current, level_width, level_height) = pyramid.last()?;
        let (next_previous, next_width, next_height) =
            downsample_half(previous, *level_width, *level_height);
        let (next_current, _, _) = downsample_half(current, *level_width, *level_height);
        pyramid.push((next_previous, next_current, next_width, next_height));
    }

    let (coarse_previous, coarse_current, coarse_width, coarse_height) = &pyramid[2];
    let mut best = (f32::MAX, 0, 0);
    let mut total = 0.0_f32;
    let mut candidates = 0_u32;
    for dy in -COARSE_RADIUS..=COARSE_RADIUS {
        for dx in -COARSE_RADIUS..=COARSE_RADIUS {
            let Some(score) = shift_error(
                coarse_previous,
                coarse_current,
                *coarse_width,
                *coarse_height,
                (dx, dy),
                1,
            ) else {
                continue;
            };
            total += score;
            candidates += 1;
            if score < best.0 {
                best = (score, dx, dy);
            }
        }
    }
    if candidates == 0 || total / candidates as f32 - best.0 < MIN_COARSE_CONTRAST {
        return None;
    }

    let mut shift = (best.1, best.2);
    let mut final_score = best.0;
    for level in [1_usize, 0] {
        let (level_previous, level_current, level_width, level_height) = &pyramid[level];
        let step = if level == 0 { 2 } else { 1 };
        let centre = (shift.0 * 2, shift.1 * 2);
        let mut level_best = (f32::MAX, centre.0, centre.1);
        for dy in -REFINE_RADIUS..=REFINE_RADIUS {
            for dx in -REFINE_RADIUS..=REFINE_RADIUS {
                let candidate = (centre.0 + dx, centre.1 + dy);
                if let Some(score) = shift_error(
                    level_previous,
                    level_current,
                    *level_width,
                    *level_height,
                    candidate,
                    step,
                ) {
                    if score < level_best.0 {
                        level_best = (score, candidate.0, candidate.1);
                    }
                }
            }
        }
        if level_best.0 == f32::MAX {
            return None;
        }
        shift = (level_best.1, level_best.2);
        final_score = level_best.0;
    }
    if final_score > MAX_ACCEPTED_ERROR {
        return None;
    }

    let (base_previous, base_current, base_width, base_height) = &pyramid[0];
    let score_at = |dx: i32, dy: i32| {
        shift_error(
            base_previous,
            base_current,
            *base_width,
            *base_height,
            (shift.0 + dx, shift.1 + dy),
            2,
        )
    };
    let fractional_x = match (score_at(-1, 0), score_at(0, 0), score_at(1, 0)) {
        (Some(before), Some(centre), Some(after)) => parabola_offset(before, centre, after),
        _ => 0.0,
    };
    let fractional_y = match (score_at(0, -1), score_at(0, 0), score_at(0, 1)) {
        (Some(before), Some(centre), Some(after)) => parabola_offset(before, centre, after),
        _ => 0.0,
    };
    Some((shift.0 as f32 + fractional_x, shift.1 as f32 + fractional_y))
}

fn camera_motion_offset(project: &Project, time: f32) -> (f32, f32) {
    if !project.camera_motion_enabled || project.camera_motion.is_empty() {
        return (0.0, 0.0);
    }
    let samples = &project.camera_motion;
    let upper = samples.partition_point(|sample| sample.time < time);
    if upper == 0 {
        return (samples[0].x, samples[0].y);
    }
    if upper >= samples.len() {
        let last = samples.last().expect("non-empty camera motion");
        return (last.x, last.y);
    }
    let before = &samples[upper - 1];
    let after = &samples[upper];
    let span = after.time - before.time;
    if span <= 0.0 {
        return (after.x, after.y);
    }
    let amount = ((time - before.time) / span).clamp(0.0, 1.0);
    (
        before.x + (after.x - before.x) * amount,
        before.y + (after.y - before.y) * amount,
    )
}

fn project_trail_points(project: &Project, time: f32) -> Vec<(f32, f32)> {
    let mut points = trail_points(
        &project.keyframes,
        time,
        project.path_smoothing,
        project.trail_duration,
        project.trail_pace,
    );
    let (offset_x, offset_y) = camera_motion_offset(project, time);
    for (x, y) in &mut points {
        *x += offset_x;
        *y += offset_y;
    }
    points
}

fn read_mjpeg_frame(
    reader: &mut impl Read,
    buffered: &mut Vec<u8>,
) -> std::io::Result<Option<Vec<u8>>> {
    const CHUNK_SIZE: usize = 64 * 1024;
    const MAX_FRAME_SIZE: usize = 32 * 1024 * 1024;
    let mut chunk = [0_u8; CHUNK_SIZE];
    loop {
        if let Some(start) = buffered.windows(2).position(|bytes| bytes == [0xFF, 0xD8]) {
            if start > 0 {
                buffered.drain(..start);
            }
            if let Some(end) = buffered
                .windows(2)
                .position(|bytes| bytes == [0xFF, 0xD9])
                .map(|index| index + 2)
            {
                return Ok(Some(buffered.drain(..end).collect()));
            }
        } else if buffered.len() > 1 {
            let keep = buffered.ends_with(&[0xFF]);
            buffered.drain(..buffered.len() - usize::from(keep));
        }

        if buffered.len() > MAX_FRAME_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "FFmpeg preview frame exceeded 32 MB",
            ));
        }
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            return Ok(None);
        }
        buffered.extend_from_slice(&chunk[..read]);
    }
}

fn export_project(
    project: &Project,
    output: &Path,
    mut report_progress: impl FnMut(f32, &str),
) -> Result<(), String> {
    report_progress(0.01, "Preparing export");
    let width = probe_video_dimension(&project.video_path, "width")?;
    let height = probe_video_dimension(&project.video_path, "height")?;
    let duration = project.trim_out - project.trim_in;
    let frames = (duration * EXPORT_FPS as f32).ceil() as u32;
    if frames == 0 || frames > FRAME_LIMIT {
        return Err("Selected trim is too long to render safely.".to_owned());
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("System clock error: {error}"))?
        .as_millis();
    let frame_dir = std::env::temp_dir().join(format!("opentracer-{nonce}"));
    fs::create_dir(&frame_dir)
        .map_err(|error| format!("Could not create render folder: {error}"))?;
    let render_result = render_overlay_frames(
        project,
        &frame_dir,
        width,
        height,
        frames,
        duration,
        &mut report_progress,
    )
    .and_then(|()| encode_export(project, output, &frame_dir, duration, &mut report_progress));
    let cleanup_result = fs::remove_dir_all(&frame_dir).map_err(|error| {
        format!(
            "Could not clean render folder {}: {error}",
            frame_dir.display()
        )
    });
    render_result?;
    cleanup_result
}

// Returns the width or height as displayed: FFmpeg auto-rotates phone footage that carries
// rotation metadata, so a quarter turn swaps the stored dimensions.
fn probe_video_dimension(path: &Path, dimension: &str) -> Result<u32, String> {
    let stdout = probe_video_stream(path)?;
    let displayed = if parse_video_rotation(&stdout) % 180 == 90 {
        if dimension == "width" {
            "height"
        } else {
            "width"
        }
    } else {
        dimension
    };
    parse_video_dimension(&stdout, displayed).ok_or_else(|| {
        format!(
            "ffprobe returned an invalid video {dimension}. Expected a positive integer; output was: {}",
            stdout.trim()
        )
    })
}

// Counter-clockwise degrees (0, 90, 180 or 270) that must be applied to show the video upright.
fn probe_video_rotation(path: &Path) -> Result<u32, String> {
    Ok(parse_video_rotation(&probe_video_stream(path)?))
}

fn probe_video_stream(path: &Path) -> Result<String, String> {
    let output = multimedia_tool_command("ffprobe")?
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=width,height:stream_side_data=rotation:stream_tags=rotate")
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .map_err(|error| format!("Could not start ffprobe: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Could not read video details: {}. ffprobe output: {}",
            String::from_utf8_lossy(&output.stderr).trim(),
            String::from_utf8_lossy(&output.stdout).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_video_rotation(output: &str) -> u32 {
    let value = |wanted: &str| {
        output.lines().find_map(|line| {
            let (key, value) = line.trim().split_once('=')?;
            let key = key.trim().trim_start_matches("TAG:");
            key.eq_ignore_ascii_case(wanted)
                .then(|| value.trim().parse::<f32>().ok())
                .flatten()
        })
    };
    // The legacy `rotate` tag is clockwise, the display matrix counter-clockwise.
    let degrees = value("rotation").or_else(|| value("rotate").map(|clockwise| -clockwise));
    let quarter_turns = (degrees.unwrap_or(0.0) / 90.0).round() as i32;
    (quarter_turns.rem_euclid(4) * 90) as u32
}

fn parse_video_dimension(output: &str, dimension: &str) -> Option<u32> {
    output.lines().find_map(|line| {
        let line = line.trim();
        let value = match line.split_once('=') {
            Some((key, value)) if key.trim() == dimension => value.trim(),
            Some(_) => return None,
            None => line,
        };
        value.parse::<u32>().ok().filter(|dimension| *dimension > 0)
    })
}

fn render_overlay_frames(
    project: &Project,
    directory: &Path,
    width: u32,
    height: u32,
    frames: u32,
    duration: f32,
    report_progress: &mut impl FnMut(f32, &str),
) -> Result<(), String> {
    if duration <= 0.0 {
        return Err("Selected trim has no duration.".to_owned());
    }
    let rgb = project.tracer.color;
    let (line_width, glow) = scaled_tracer_dimensions(&project.tracer, height as f32);
    let animated_masks = project
        .masks
        .iter()
        .any(|region| !region.keyframes.is_empty());
    let static_coverage = if animated_masks {
        None
    } else {
        mask_coverage_image(width, height, &project.masks, 0.0)
    };
    let render_frame = |frame_index: u32| -> Result<(), String> {
        let time = project.trim_in + frame_index as f32 / EXPORT_FPS as f32;
        let points = project_trail_points(project, time);
        let mut image = RgbaImage::new(width, height);
        let line_color = Rgba([rgb[0], rgb[1], rgb[2], 255]);
        let scaled: Vec<(f32, f32)> = points
            .iter()
            .map(|(x, y)| (x * width as f32, y * height as f32))
            .collect();
        draw_feathered_path(
            &mut image,
            &scaled,
            line_width,
            glow,
            rgb,
            project.tracer.glow_color,
        );
        if let Some((x, y)) = scaled.last().copied() {
            draw_filled_circle_mut(
                &mut image,
                (x.round() as i32, y.round() as i32),
                line_width.ceil() as i32,
                line_color,
            );
        }
        let animated_coverage = animated_masks
            .then(|| mask_coverage_image(width, height, &project.masks, time))
            .flatten();
        if let Some(coverage) = static_coverage.as_ref().or(animated_coverage.as_ref()) {
            apply_mask_coverage(&mut image, coverage);
        }
        let frame_path = directory.join(format!("overlay_{frame_index:06}.png"));
        let file = fs::File::create(&frame_path)
            .map_err(|error| format!("Could not render tracer frame: {error}"))?;
        // Frames are temporary, so favour encoding speed over file size.
        image
            .write_with_encoder(PngEncoder::new_with_quality(
                BufWriter::new(file),
                CompressionType::Fast,
                PngFilterType::Sub,
            ))
            .map_err(|error| format!("Could not render tracer frame: {error}"))
    };

    let next_frame = AtomicU32::new(0);
    let completed_frames = AtomicU32::new(0);
    let failed = AtomicBool::new(false);
    let failure = Mutex::new(None::<String>);
    let worker_count = std::thread::available_parallelism()
        .map_or(2, |count| count.get())
        .clamp(1, 8)
        .min(frames as usize);
    let worker = || loop {
        if failed.load(Ordering::Relaxed) {
            break;
        }
        let frame_index = next_frame.fetch_add(1, Ordering::Relaxed);
        if frame_index >= frames {
            break;
        }
        match render_frame(frame_index) {
            Ok(()) => {
                completed_frames.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert(error);
                failed.store(true, Ordering::Relaxed);
                break;
            }
        }
    };
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..worker_count).map(|_| scope.spawn(worker)).collect();
        while !handles.iter().all(|handle| handle.is_finished()) {
            let progress = completed_frames.load(Ordering::Relaxed) as f32 / frames as f32;
            report_progress(0.02 + progress * 0.48, "Rendering tracer");
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    if let Some(error) = failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(error);
    }
    report_progress(0.50, "Rendering tracer");
    Ok(())
}

fn draw_thick_line(
    image: &mut RgbaImage,
    start: (f32, f32),
    end: (f32, f32),
    width: u32,
    color: Rgba<u8>,
) {
    draw_line_segment_mut(image, start, end, color);
    let radius = (width / 2) as i32;
    if radius == 0 {
        return;
    }
    let distance = ((end.0 - start.0).powi(2) + (end.1 - start.1).powi(2)).sqrt();
    let steps = (distance / (radius as f32).max(1.0)).ceil().max(1.0) as u32;
    for step in 0..=steps {
        let amount = step as f32 / steps as f32;
        let x = start.0 + (end.0 - start.0) * amount;
        let y = start.1 + (end.1 - start.1) * amount;
        draw_filled_circle_mut(image, (x.round() as i32, y.round() as i32), radius, color);
    }
}

fn polygon_area(points: &[(f32, f32)]) -> f32 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(a, b)| a.0 * b.1 - b.0 * a.1)
        .sum::<f32>()
        * 0.5
}

// Ear-clipping triangulation, so concave mask shapes can be filled with a mesh.
fn triangulate_polygon(points: &[(f32, f32)]) -> Vec<[usize; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }
    let mut remaining: Vec<usize> = (0..points.len()).collect();
    if polygon_area(points) < 0.0 {
        remaining.reverse();
    }
    let cross = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };
    let mut triangles = Vec::new();
    while remaining.len() > 3 {
        let count = remaining.len();
        let ear = (0..count).find(|&i| {
            let (a, b, c) = (
                remaining[(i + count - 1) % count],
                remaining[i],
                remaining[(i + 1) % count],
            );
            if cross(points[a], points[b], points[c]) <= 0.0 {
                return false;
            }
            !remaining.iter().any(|&other| {
                other != a
                    && other != b
                    && other != c
                    && cross(points[a], points[b], points[other]) >= 0.0
                    && cross(points[b], points[c], points[other]) >= 0.0
                    && cross(points[c], points[a], points[other]) >= 0.0
            })
        });
        // Degenerate or self-intersecting shapes: clip an arbitrary vertex to guarantee progress.
        let i = ear.unwrap_or(0);
        triangles.push([
            remaining[(i + count - 1) % count],
            remaining[i],
            remaining[(i + 1) % count],
        ]);
        remaining.remove(i);
    }
    triangles.push([remaining[0], remaining[1], remaining[2]]);
    triangles
}

// Coverage mask (255 inside any region) with a one pixel soft edge.
fn mask_coverage_image(
    width: u32,
    height: u32,
    masks: &[MaskRegion],
    time: f32,
) -> Option<GrayImage> {
    let mut coverage = GrayImage::new(width, height);
    let mut drawn = false;
    for region in masks {
        let mut polygon: Vec<imageproc::point::Point<i32>> = region
            .shape_at(time)
            .iter()
            .map(|(x, y)| {
                imageproc::point::Point::new(
                    (x * width as f32).round() as i32,
                    (y * height as f32).round() as i32,
                )
            })
            .collect();
        polygon.dedup();
        while polygon.len() > 1 && polygon.first() == polygon.last() {
            polygon.pop();
        }
        if polygon.len() >= 3 {
            imageproc::drawing::draw_polygon_mut(&mut coverage, &polygon, Luma([255]));
            drawn = true;
        }
    }
    drawn.then(|| box_filter(&coverage, 1, 1))
}

fn apply_mask_coverage(image: &mut RgbaImage, coverage: &GrayImage) {
    for (pixel, cover) in image.pixels_mut().zip(coverage.pixels()) {
        let cover = cover.0[0] as u32;
        if cover > 0 {
            pixel.0[3] = (pixel.0[3] as u32 * (255 - cover) / 255) as u8;
        }
    }
}

fn render_glow_layer(
    width: u32,
    height: u32,
    points: &[(f32, f32)],
    core_width: f32,
    glow: f32,
    rgb: [u8; 3],
) -> Option<RgbaImage> {
    if glow <= 0.0 || points.len() < 2 {
        return None;
    }
    Some(glow_pixels(width, height, points, core_width, glow, rgb))
}

// Like `render_glow_layer`, but only covers the glow's bounding box so large frames stay cheap.
// Returns the layer with the position of its top-left corner in the full frame.
fn render_glow_region(
    width: u32,
    height: u32,
    points: &[(f32, f32)],
    core_width: f32,
    glow: f32,
    rgb: [u8; 3],
) -> Option<(RgbaImage, i64, i64)> {
    if glow <= 0.0 || points.len() < 2 {
        return None;
    }
    let stroke_radius = (core_width + glow * 2.0).ceil().max(1.0) / 2.0;
    let blur_extent = 3.0 * (glow * 0.32).round().max(1.0);
    let margin = stroke_radius + blur_extent + 4.0;
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for (x, y) in points {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }
    let left = (min_x - margin).floor().clamp(0.0, width as f32) as u32;
    let top = (min_y - margin).floor().clamp(0.0, height as f32) as u32;
    let right = (max_x + margin).ceil().clamp(0.0, width as f32) as u32;
    let bottom = (max_y + margin).ceil().clamp(0.0, height as f32) as u32;
    if right <= left || bottom <= top {
        return None;
    }
    let local: Vec<(f32, f32)> = points
        .iter()
        .map(|(x, y)| (x - left as f32, y - top as f32))
        .collect();
    Some((
        glow_pixels(right - left, bottom - top, &local, core_width, glow, rgb),
        left as i64,
        top as i64,
    ))
}

fn glow_pixels(
    width: u32,
    height: u32,
    points: &[(f32, f32)],
    core_width: f32,
    glow: f32,
    rgb: [u8; 3],
) -> RgbaImage {
    let mut mask = GrayImage::new(width, height);
    let stroke_width = (core_width + glow * 2.0).ceil().max(1.0) as u32;
    for segment in points.windows(2) {
        draw_glow_mask_line(&mut mask, segment[0], segment[1], stroke_width);
    }

    let blur_radius = (glow * 0.32).round().max(1.0) as u32;
    let first_blur = box_filter(&mask, blur_radius, blur_radius);
    let second_blur = box_filter(&first_blur, blur_radius, blur_radius);
    let blurred_mask = box_filter(&second_blur, blur_radius, blur_radius);
    let glow_strength = 0.28;
    RgbaImage::from_fn(width, height, |x, y| {
        let alpha = (blurred_mask.get_pixel(x, y).0[0] as f32 * glow_strength).round() as u8;
        Rgba([rgb[0], rgb[1], rgb[2], alpha])
    })
}

fn draw_glow_mask_line(mask: &mut GrayImage, start: (f32, f32), end: (f32, f32), width: u32) {
    let color = Luma([255]);
    draw_line_segment_mut(mask, start, end, color);
    let radius = (width / 2) as i32;
    if radius == 0 {
        return;
    }
    let distance = ((end.0 - start.0).powi(2) + (end.1 - start.1).powi(2)).sqrt();
    let steps = (distance / (radius as f32).max(1.0)).ceil().max(1.0) as u32;
    for step in 0..=steps {
        let amount = step as f32 / steps as f32;
        let x = start.0 + (end.0 - start.0) * amount;
        let y = start.1 + (end.1 - start.1) * amount;
        draw_filled_circle_mut(mask, (x.round() as i32, y.round() as i32), radius, color);
    }
}

fn draw_feathered_path(
    image: &mut RgbaImage,
    points: &[(f32, f32)],
    core_width: f32,
    glow: f32,
    line_rgb: [u8; 3],
    glow_rgb: [u8; 3],
) {
    if let Some((glow_layer, offset_x, offset_y)) = render_glow_region(
        image.width(),
        image.height(),
        points,
        core_width,
        glow,
        glow_rgb,
    ) {
        image::imageops::overlay(image, &glow_layer, offset_x, offset_y);
    }
    for segment in points.windows(2) {
        draw_thick_line(
            image,
            segment[0],
            segment[1],
            core_width.ceil() as u32,
            Rgba([line_rgb[0], line_rgb[1], line_rgb[2], 255]),
        );
    }
}

fn trail_points(
    keys: &[Keyframe],
    time: f32,
    smoothing: f32,
    trail_duration: f32,
    trail_pace: f32,
) -> Vec<(f32, f32)> {
    const CURVE_SAMPLES: usize = 16;
    let Some(first) = keys.first() else {
        return Vec::new();
    };
    if time < first.time {
        return Vec::new();
    }
    let smoothed_keys = smooth_keyframes(keys, smoothing);
    let keys = smoothed_keys.as_slice();
    if keys.len() == 1 || time <= keys[0].time {
        return vec![(keys[0].x, keys[0].y)];
    }

    let window_start = trail_cutoff_time(keys[0].time, time, trail_duration, trail_pace);
    let mut points = Vec::new();
    for segment_index in 0..keys.len().saturating_sub(1) {
        let start = &keys[segment_index];
        let end = &keys[segment_index + 1];
        if end.time <= window_start {
            continue;
        }
        let span = (end.time - start.time).max(f32::EPSILON);
        let start_fraction = ((window_start - start.time) / span).clamp(0.0, 1.0);
        let end_fraction = ((time - start.time) / span).clamp(0.0, 1.0);
        let p0 = &keys[segment_index.saturating_sub(1)];
        let p3 = &keys[(segment_index + 2).min(keys.len() - 1)];
        if end_fraction < start_fraction {
            break;
        }
        if points.is_empty() {
            points.push(catmull_rom_point(p0, start, end, p3, start_fraction));
        }
        for sample in 1..=CURVE_SAMPLES {
            let sample_fraction = sample as f32 / CURVE_SAMPLES as f32;
            if sample_fraction <= start_fraction {
                continue;
            }
            if sample_fraction >= end_fraction {
                break;
            }
            points.push(catmull_rom_point(p0, start, end, p3, sample_fraction));
        }
        if end_fraction > start_fraction {
            points.push(catmull_rom_point(p0, start, end, p3, end_fraction));
        }
        if end_fraction < 1.0 {
            break;
        }
    }
    if points.is_empty() {
        // The trail has fully caught up with the final keyframe, so the end point goes too.
        if window_start >= keys[keys.len() - 1].time && time > window_start {
            return points;
        }
        let (x, y) = position_on_path(keys, time);
        points.push((x, y));
    }
    points
}

fn position_on_path(keys: &[Keyframe], time: f32) -> (f32, f32) {
    if time <= keys[0].time {
        return (keys[0].x, keys[0].y);
    }
    for segment_index in 0..keys.len().saturating_sub(1) {
        let start = &keys[segment_index];
        let end = &keys[segment_index + 1];
        if time <= end.time {
            let span = (end.time - start.time).max(f32::EPSILON);
            let fraction = ((time - start.time) / span).clamp(0.0, 1.0);
            let p0 = &keys[segment_index.saturating_sub(1)];
            let p3 = &keys[(segment_index + 2).min(keys.len() - 1)];
            return catmull_rom_point(p0, start, end, p3, fraction);
        }
    }
    let last = keys.last().expect("non-empty keyframes");
    (last.x, last.y)
}

fn smooth_keyframes(keys: &[Keyframe], amount: f32) -> Vec<Keyframe> {
    let mut smoothed = keys.to_vec();
    let strength = amount.clamp(0.0, 1.0);
    let blend = strength * 0.25;
    for _ in 0..4 {
        let previous = smoothed.clone();
        for index in 1..smoothed.len().saturating_sub(1) {
            let average_x = (previous[index - 1].x + previous[index + 1].x) * 0.5;
            let average_y = (previous[index - 1].y + previous[index + 1].y) * 0.5;
            smoothed[index].x = previous[index].x * (1.0 - blend) + average_x * blend;
            smoothed[index].y = previous[index].y * (1.0 - blend) + average_y * blend;
        }
    }
    smoothed
}

fn catmull_rom_point(
    p0: &Keyframe,
    p1: &Keyframe,
    p2: &Keyframe,
    p3: &Keyframe,
    t: f32,
) -> (f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    let interpolate = |a: f32, b: f32, c: f32, d: f32| {
        0.5 * ((2.0 * b)
            + (-a + c) * t
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (-a + 3.0 * b - 3.0 * c + d) * t3)
    };
    (
        interpolate(p0.x, p1.x, p2.x, p3.x),
        interpolate(p0.y, p1.y, p2.y, p3.y),
    )
}

// Encoder argument sets in order of preference; GPU encoders first, software last.
fn export_encoder_candidates() -> Vec<Vec<&'static str>> {
    if cfg!(target_os = "macos") {
        return vec![vec![
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "10M",
            "-allow_sw",
            "1",
        ]];
    }
    vec![
        vec![
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "21",
            "-b:v",
            "0",
        ],
        vec![
            "-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22",
        ],
        vec!["-c:v", "h264_qsv", "-global_quality", "21"],
        vec!["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"],
    ]
}

fn encode_export(
    project: &Project,
    output: &Path,
    frame_dir: &Path,
    duration: f32,
    report_progress: &mut impl FnMut(f32, &str),
) -> Result<(), String> {
    let candidates = export_encoder_candidates();
    let mut last_error = String::new();
    for encoder in candidates {
        match encode_export_with(
            project,
            output,
            frame_dir,
            duration,
            &encoder,
            report_progress,
        ) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

fn encode_export_with(
    project: &Project,
    output: &Path,
    frame_dir: &Path,
    duration: f32,
    encoder: &[&str],
    report_progress: &mut impl FnMut(f32, &str),
) -> Result<(), String> {
    let rotation = probe_video_rotation(&project.video_path)?;
    // FFmpeg would copy the source's rotation flag onto the already-upright output, so the
    // flag is cleared on input and the rotation is applied explicitly instead.
    let upright = match rotation {
        90 => "transpose=2",
        180 => "hflip,vflip",
        270 => "transpose=1",
        _ => "null",
    };
    let filter = format!("[0:v]{upright}[base];[base][1:v]overlay=shortest=1:format=auto[outv]");
    let pattern = frame_dir.join("overlay_%06d.png");
    let mut command = multimedia_tool_command("ffmpeg")?;
    command.args(["-y", "-v", "error", "-ss"]);
    command.arg(format!("{:.3}", project.trim_in));
    if rotation != 0 {
        command.args(["-display_rotation", "0"]);
    }
    command
        .arg("-i")
        .arg(&project.video_path)
        .arg("-t")
        .arg(format!("{duration:.3}"))
        .arg("-framerate")
        .arg(EXPORT_FPS.to_string())
        .arg("-i")
        .arg(pattern)
        .args(["-filter_complex", &filter, "-map", "[outv]", "-map", "0:a?"]);
    command.args(encoder);
    let mut child = command
        .args([
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-progress",
            "pipe:1",
            "-nostats",
        ])
        .arg(output)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not start FFmpeg export: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Could not read FFmpeg export progress.".to_owned())?;
    let reader = BufReader::new(stdout);
    let mut last_report = Instant::now();
    for line in reader.lines() {
        let line =
            line.map_err(|error| format!("Could not read FFmpeg export progress: {error}"))?;
        if let Some(value) = line.strip_prefix("out_time_us=") {
            if let Ok(microseconds) = value.parse::<f64>() {
                let fraction =
                    (microseconds / (duration as f64 * 1_000_000.0)).clamp(0.0, 1.0) as f32;
                if last_report.elapsed() >= Duration::from_millis(100) || fraction >= 1.0 {
                    report_progress(0.50 + fraction * 0.49, "Encoding MP4");
                    last_report = Instant::now();
                }
            }
        }
    }
    let result = child
        .wait_with_output()
        .map_err(|error| format!("Could not finish FFmpeg export: {error}"))?;
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).trim().to_owned());
    }
    report_progress(0.99, "Finishing export");
    Ok(())
}

fn main() -> eframe::Result<()> {
    let app_icon = image::load_from_memory(include_bytes!("../assets/branding/app_icon.png"))
        .expect("embedded app icon must be a valid image")
        .to_rgba8();
    let app_icon =
        image::imageops::resize(&app_icon, 256, 256, image::imageops::FilterType::Lanczos3);
    let app_icon = egui::IconData {
        rgba: app_icon.into_raw(),
        width: 256,
        height: 256,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_icon(app_icon),
        ..Default::default()
    };
    eframe::run_native(
        "Tommy'sTracer - Easy Golf Tracer",
        options,
        Box::new(|creation_context| Ok(Box::new(TracerApp::new(&creation_context.egui_ctx)))),
    )
}

fn apply_app_theme(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 9.0);
    style.spacing.button_padding = Vec2::new(12.0, 7.0);
    style.spacing.interact_size = Vec2::new(34.0, 32.0);

    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(Color32::from_rgb(228, 236, 246));
    visuals.panel_fill = Color32::from_rgb(19, 29, 41);
    visuals.window_fill = Color32::from_rgb(29, 43, 59);
    visuals.window_stroke = Stroke::new(1.0_f32, Color32::from_white_alpha(38));
    visuals.window_corner_radius = egui::CornerRadius::same(14);
    visuals.faint_bg_color = Color32::from_rgb(37, 53, 71);
    visuals.extreme_bg_color = Color32::from_rgb(12, 19, 28);
    visuals.code_bg_color = Color32::from_rgb(22, 34, 47);
    visuals.hyperlink_color = Color32::from_rgb(93, 177, 255);
    visuals.selection.bg_fill = Color32::from_rgb(43, 132, 255);
    visuals.selection.stroke = Stroke::new(1.0_f32, Color32::WHITE);

    let mut noninteractive = visuals.widgets.noninteractive;
    noninteractive.bg_fill = Color32::from_rgb(25, 38, 52);
    noninteractive.weak_bg_fill = Color32::from_rgb(25, 38, 52);
    noninteractive.bg_stroke = Stroke::new(1.0_f32, Color32::from_white_alpha(28));
    noninteractive.fg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(218, 229, 241));
    noninteractive.corner_radius = egui::CornerRadius::same(10);
    visuals.widgets.noninteractive = noninteractive;

    let mut inactive = visuals.widgets.inactive;
    inactive.bg_fill = Color32::from_rgb(43, 59, 77);
    inactive.weak_bg_fill = Color32::from_rgb(43, 59, 77);
    inactive.bg_stroke = Stroke::new(1.0_f32, Color32::from_white_alpha(28));
    inactive.fg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(228, 236, 246));
    inactive.corner_radius = egui::CornerRadius::same(9);
    visuals.widgets.inactive = inactive;

    let mut hovered = inactive;
    hovered.bg_fill = Color32::from_rgb(57, 82, 108);
    hovered.weak_bg_fill = Color32::from_rgb(57, 82, 108);
    hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(105, 177, 255));
    visuals.widgets.hovered = hovered;

    let mut active = hovered;
    active.bg_fill = Color32::from_rgb(43, 132, 255);
    active.weak_bg_fill = Color32::from_rgb(43, 132, 255);
    active.fg_stroke = Stroke::new(1.0_f32, Color32::WHITE);
    visuals.widgets.active = active;
    visuals.widgets.open = hovered;
    style.visuals = visuals;
    ctx.set_style(style);
}

#[cfg(test)]
mod tests {
    use super::{
        camera_motion_offset, default_trail_duration, default_trail_pace, draw_feathered_path,
        ensure_keyframe_at_time, estimate_frame_translation, expected_motion_analysis_samples,
        frame_time, internal_to_style_scale, internal_to_style_scale_from, keyframe_index_at_time,
        motion_analysis_fps, parse_frame_rate, parse_video_dimension, parse_video_rotation,
        playback_frame_deadline, position_on_path, preview_tracer_zoom_scale, project_trail_points,
        read_mjpeg_frame, render_glow_layer, render_glow_region, scaled_tracer_dimensions,
        smooth_keyframes, snap_trim_time, step_frame, style_scale_to_internal,
        style_scale_to_internal_from, time_to_frame, timeline_range, trace_nudge_pixels,
        trail_cutoff_time, trail_pace_index, trail_points, zoom_pan_for_cursor,
        zoomed_preview_rect, CameraMotionSample, FrameCache, Keyframe, Project, TracerStyle,
        MAX_CANVAS_ZOOM, TRACER_GLOW_MAX, TRACER_WIDTH_MAX,
    };
    use eframe::egui::{Pos2, Rect, Vec2};
    use image::{Rgba, RgbaImage};
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[test]
    fn frame_rate_parser_accepts_fractional_and_decimal_values() {
        assert_eq!(parse_frame_rate("30000/1001"), Some(30000.0 / 1001.0));
        assert_eq!(parse_frame_rate("29.97"), Some(29.97));
        assert_eq!(parse_frame_rate("0/0"), None);
        assert_eq!(parse_frame_rate("N/A"), None);
    }

    #[test]
    fn video_dimension_parser_reads_named_or_plain_positive_values() {
        assert_eq!(parse_video_dimension("width=1920\n", "width"), Some(1920));
        assert_eq!(parse_video_dimension("1080\n", "height"), Some(1080));
        assert_eq!(
            parse_video_dimension("height=1080\nwidth=1920\n", "width"),
            Some(1920)
        );
        assert_eq!(parse_video_dimension("width=N/A\n", "width"), None);
        assert_eq!(parse_video_dimension("width=0\n", "width"), None);
    }

    #[test]
    fn zero_glow_renders_no_glow_layer() {
        let points = [(10.0, 10.0), (60.0, 40.0)];
        assert!(render_glow_layer(100, 80, &points, 4.0, 0.0, [255, 0, 0]).is_none());
        assert!(render_glow_region(100, 80, &points, 4.0, 0.0, [255, 0, 0]).is_none());
    }

    #[test]
    fn tracer_style_controls_map_whole_number_scale_to_internal_values() {
        assert_eq!(style_scale_to_internal(1, TRACER_WIDTH_MAX), 1.6);
        assert_eq!(
            style_scale_to_internal(10, TRACER_WIDTH_MAX),
            TRACER_WIDTH_MAX
        );
        assert_eq!(style_scale_to_internal(1, TRACER_GLOW_MAX), 2.4);
        assert_eq!(
            style_scale_to_internal(10, TRACER_GLOW_MAX),
            TRACER_GLOW_MAX
        );
        assert_eq!(style_scale_to_internal(5, 1.0), 0.5);
        assert_eq!(internal_to_style_scale(5.0, TRACER_WIDTH_MAX), 3);
        assert_eq!(internal_to_style_scale(10.0, TRACER_GLOW_MAX), 4);
        assert_eq!(internal_to_style_scale(0.5, 1.0), 5);
        assert_eq!(style_scale_to_internal_from(0, 0, TRACER_GLOW_MAX), 0.0);
        assert_eq!(internal_to_style_scale_from(0, 0.0, TRACER_GLOW_MAX), 0);
        assert_eq!(internal_to_style_scale(0.0, TRACER_GLOW_MAX), 1);
    }

    #[test]
    fn tracer_and_glow_colors_default_to_the_same_red() {
        let style = TracerStyle::default();
        assert_eq!(style.color, [255, 76, 48]);
        assert_eq!(style.glow_color, style.color);
    }

    #[test]
    fn wasd_nudge_step_changes_after_four_x_zoom_and_shift_stays_fine() {
        assert_eq!(trace_nudge_pixels(1.0, false), 10.0);
        assert_eq!(trace_nudge_pixels(4.0, false), 10.0);
        assert_eq!(trace_nudge_pixels(4.01, false), 5.0);
        assert_eq!(trace_nudge_pixels(8.0, false), 5.0);
        assert_eq!(trace_nudge_pixels(8.0, true), 1.0);
    }

    #[test]
    fn trail_pace_controls_how_long_the_configured_trail_remains_visible() {
        assert_eq!(trail_pace_index(0.25), 0);
        assert_eq!(trail_pace_index(0.5), 1);
        assert_eq!(trail_pace_index(1.0), 2);
        assert_eq!(trail_cutoff_time(0.0, 5.0, 5.0, 0.25), 0.0);
        assert_eq!(trail_cutoff_time(0.0, 5.0, 5.0, 0.5), 0.0);
        assert_eq!(trail_cutoff_time(0.0, 5.0, 5.0, 1.0), 0.0);
        assert_eq!(trail_cutoff_time(0.0, 9.0, 5.0, 0.25), 1.0);
        assert_eq!(trail_cutoff_time(0.0, 9.0, 5.0, 0.5), 2.0);
        assert_eq!(trail_cutoff_time(0.0, 9.0, 5.0, 1.0), 4.0);
        assert_eq!(default_trail_pace(), 0.5);
    }

    #[test]
    fn preview_frame_deadlines_follow_the_source_clock() {
        let start = Instant::now();
        assert_eq!(playback_frame_deadline(start, 0, 25.0), start);
        assert_eq!(
            playback_frame_deadline(start, 5, 25.0).duration_since(start),
            Duration::from_millis(200)
        );
    }

    #[test]
    fn tracing_timeline_is_limited_to_the_trimmed_range() {
        assert_eq!(timeline_range(95.2, 36.04, 43.88, false), (36.04, 43.88));
        assert_eq!(timeline_range(95.2, 36.04, 43.88, true), (0.0, 95.2));
    }

    #[test]
    fn tracer_dimensions_scale_consistently_between_preview_and_export() {
        let style = TracerStyle {
            width: 8.0,
            glow: 12.0,
            ..TracerStyle::default()
        };
        let (preview_width, preview_glow) = scaled_tracer_dimensions(&style, 540.0);
        let (export_width, export_glow) = scaled_tracer_dimensions(&style, 2160.0);
        assert_eq!((preview_width, preview_glow), (4.0, 6.0));
        assert_eq!((export_width, export_glow), (16.0, 24.0));
    }

    #[test]
    fn preview_zoom_keeps_the_pointer_anchor_fixed() {
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 450.0));
        let image = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 450.0));
        let pointer = Pos2::new(600.0, 300.0);
        let pan = zoom_pan_for_cursor(canvas, image, image.size(), pointer, 2.0);
        let zoomed = zoomed_preview_rect(canvas, image.size(), 2.0, pan);
        let anchor = Pos2::new(
            zoomed.left() + 0.75 * zoomed.width(),
            zoomed.top() + (2.0 / 3.0) * zoomed.height(),
        );
        assert!((anchor.x - pointer.x).abs() < 0.001);
        assert!((anchor.y - pointer.y).abs() < 0.001);
    }

    #[test]
    fn preview_zoom_and_pan_stay_within_supported_bounds() {
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 450.0));
        let image = zoomed_preview_rect(
            canvas,
            Vec2::new(800.0, 450.0),
            MAX_CANVAS_ZOOM + 1.0,
            Vec2::new(10_000.0, -10_000.0),
        );
        assert_eq!(image.width(), 800.0 * MAX_CANVAS_ZOOM);
        assert_eq!(image.left(), canvas.left());
        assert_eq!(image.bottom(), canvas.bottom());
    }

    #[test]
    fn preview_tracer_thickness_and_glow_shrink_gently_with_zoom() {
        assert_eq!(preview_tracer_zoom_scale(1.0), 1.0);
        let four_x_size = 4.0 * preview_tracer_zoom_scale(4.0);
        let eight_x_size = 8.0 * preview_tracer_zoom_scale(8.0);
        assert!(four_x_size < 1.0);
        assert!(eight_x_size < four_x_size);
        assert!(eight_x_size > 0.7);
    }

    #[test]
    fn rotation_metadata_is_normalised_and_swaps_displayed_dimensions() {
        assert_eq!(parse_video_rotation("width=1920\nrotation=-90\n"), 270);
        assert_eq!(parse_video_rotation("width=1920\nrotation=90\n"), 90);
        assert_eq!(parse_video_rotation("width=1920\nrotation=-180\n"), 180);
        assert_eq!(parse_video_rotation("width=1920\nTAG:rotate=90\n"), 270);
        assert_eq!(parse_video_rotation("width=1920\nTAG:rotate=270\n"), 90);
        assert_eq!(parse_video_rotation("width=1920\nheight=1080\n"), 0);
    }

    #[test]
    fn tracer_is_hidden_before_first_keyframe() {
        let keys = [Keyframe {
            time: 1.0,
            x: 0.2,
            y: 0.4,
        }];
        assert!(trail_points(&keys, 0.5, 0.0, 5.0, 1.0).is_empty());
    }

    #[test]
    fn current_tracer_position_is_interpolated_between_keyframes() {
        let keys = [
            Keyframe {
                time: 0.0,
                x: 0.2,
                y: 0.4,
            },
            Keyframe {
                time: 2.0,
                x: 0.8,
                y: 0.6,
            },
        ];
        let points = trail_points(&keys, 1.0, 0.0, 5.0, 1.0);
        let (x, y) = points.last().copied().expect("interpolated point");
        assert!((x - 0.5).abs() < 0.0001);
        assert!((y - 0.5).abs() < 0.0001);
    }

    #[test]
    fn tracer_path_curves_smoothly_through_keyframes() {
        let keys = [
            Keyframe {
                time: 0.0,
                x: 0.1,
                y: 0.0,
            },
            Keyframe {
                time: 1.0,
                x: 0.3,
                y: 0.5,
            },
            Keyframe {
                time: 2.0,
                x: 0.7,
                y: 0.5,
            },
            Keyframe {
                time: 3.0,
                x: 0.9,
                y: 0.0,
            },
        ];
        let points = trail_points(&keys, 1.5, 0.0, 5.0, 1.0);
        assert!(points.iter().any(|(_, y)| *y > 0.5));
        let (x, y) = points.last().copied().expect("interpolated point");
        assert!((x - 0.5).abs() < 0.0001);
        assert!((y - 0.5625).abs() < 0.0001);
    }

    #[test]
    fn final_point_disappears_once_the_trail_has_fully_faded() {
        let keys = [
            Keyframe {
                time: 0.0,
                x: 0.1,
                y: 0.2,
            },
            Keyframe {
                time: 1.0,
                x: 0.9,
                y: 0.8,
            },
        ];
        assert!(!trail_points(&keys, 1.5, 0.0, 1.0, 1.0).is_empty());
        assert!(trail_points(&keys, 2.5, 0.0, 1.0, 1.0).is_empty());
    }

    #[test]
    fn trail_duration_clips_the_oldest_path_at_the_exact_time_cutoff() {
        let keys = [
            Keyframe {
                time: 0.0,
                x: 0.1,
                y: 0.2,
            },
            Keyframe {
                time: 1.0,
                x: 0.4,
                y: 0.6,
            },
            Keyframe {
                time: 2.0,
                x: 0.9,
                y: 0.8,
            },
        ];
        let points = trail_points(&keys, 2.0, 0.0, 1.0, 1.0);
        assert_eq!(points.first().copied(), Some((0.4, 0.6)));
        let last = points.last().copied().expect("current tracer point");
        assert!((last.0 - 0.9).abs() < 0.0001);
        assert!((last.1 - 0.8).abs() < 0.0001);
        assert!(points.iter().all(|(x, _)| *x >= 0.4));
    }

    #[test]
    fn trail_duration_cutoff_interpolates_between_keyframes() {
        let keys = [
            Keyframe {
                time: 0.0,
                x: 0.1,
                y: 0.2,
            },
            Keyframe {
                time: 2.0,
                x: 0.9,
                y: 0.8,
            },
        ];
        let points = trail_points(&keys, 2.0, 0.0, 1.0, 1.0);
        let expected = position_on_path(&keys, 1.0);
        assert!((points[0].0 - expected.0).abs() < 0.0001);
        assert!((points[0].1 - expected.1).abs() < 0.0001);
        let last = points.last().copied().expect("current tracer point");
        assert!((last.0 - 0.9).abs() < 0.0001);
        assert!((last.1 - 0.8).abs() < 0.0001);
    }

    #[test]
    fn project_trail_pace_changes_the_cutoff_speed_after_the_duration() {
        let mut project = Project {
            video_path: PathBuf::new(),
            duration: 3.0,
            frame_rate: 30.0,
            trim_in: 0.0,
            trim_out: 3.0,
            keyframes: vec![
                Keyframe {
                    time: 0.0,
                    x: 0.1,
                    y: 0.2,
                },
                Keyframe {
                    time: 2.0,
                    x: 0.9,
                    y: 0.8,
                },
            ],
            camera_motion: Vec::new(),
            camera_motion_enabled: false,
            masks: Vec::new(),
            ball_colour: None,
            path_smoothing: 0.0,
            trail_duration: 1.0,
            trail_pace: 0.5,
            tracer: TracerStyle::default(),
        };
        let half_speed_points = project_trail_points(&project, 2.0);
        let half_speed_cutoff = half_speed_points.first().copied().expect("trail cutoff");
        let expected_half_speed = position_on_path(&project.keyframes, 0.5);
        assert!((half_speed_cutoff.0 - expected_half_speed.0).abs() < 0.0001);
        assert!((half_speed_cutoff.1 - expected_half_speed.1).abs() < 0.0001);

        project.trail_pace = 1.0;
        let full_speed_points = project_trail_points(&project, 2.0);
        let full_speed_cutoff = full_speed_points.first().copied().expect("trail cutoff");
        let expected_full_speed = position_on_path(&project.keyframes, 1.0);
        assert!((full_speed_cutoff.0 - expected_full_speed.0).abs() < 0.0001);
        assert!((full_speed_cutoff.1 - expected_full_speed.1).abs() < 0.0001);
        assert!(half_speed_cutoff.0 < full_speed_cutoff.0);
    }

    // Smooth blobs give the matcher realistic, non-repeating texture at any sub-pixel offset.
    fn blob_scene(width: u32, height: u32, shift_x: f32, shift_y: f32) -> Vec<u8> {
        const BLOBS: [(f32, f32, f32, f32); 7] = [
            (0.20, 0.25, 0.10, 90.0),
            (0.70, 0.30, 0.07, -70.0),
            (0.45, 0.60, 0.12, 110.0),
            (0.80, 0.75, 0.06, -90.0),
            (0.15, 0.80, 0.08, 80.0),
            (0.55, 0.15, 0.05, 100.0),
            (0.90, 0.50, 0.09, -60.0),
        ];
        let mut pixels = vec![0_u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let (sx, sy) = (x as f32 - shift_x, y as f32 - shift_y);
                let mut value = 128.0;
                for (cx, cy, radius, amplitude) in BLOBS {
                    let dx = sx / width as f32 - cx;
                    let dy = sy / width as f32 - cy;
                    value += amplitude * (-(dx * dx + dy * dy) / (radius * radius)).exp();
                }
                pixels[(y * width + x) as usize] = value.clamp(0.0, 255.0) as u8;
            }
        }
        pixels
    }

    #[test]
    fn camera_motion_estimator_detects_small_scene_translations() {
        let (width, height) = (240_u32, 160_u32);
        let previous = blob_scene(width, height, 0.0, 0.0);
        let current = blob_scene(width, height, 3.0, -2.0);
        let (x, y) = estimate_frame_translation(&previous, &current, width, height)
            .expect("textured frames should match");
        assert!((x - 3.0).abs() < 0.6 && (y + 2.0).abs() < 0.6, "({x}, {y})");
    }

    #[test]
    fn camera_motion_estimator_follows_fast_camera_moves() {
        let (width, height) = (240_u32, 160_u32);
        let previous = blob_scene(width, height, 0.0, 0.0);
        let current = blob_scene(width, height, -22.0, 15.0);
        let (x, y) = estimate_frame_translation(&previous, &current, width, height)
            .expect("fast moves should still match");
        assert!(
            (x + 22.0).abs() < 1.5 && (y - 15.0).abs() < 1.5,
            "({x}, {y})"
        );
    }

    #[test]
    fn camera_motion_estimator_rejects_featureless_frames() {
        let flat = vec![100_u8; 240 * 160];
        assert!(estimate_frame_translation(&flat, &flat, 240, 160).is_none());
    }

    #[test]
    fn motion_analysis_reports_samples_for_the_selected_duration() {
        assert_eq!(expected_motion_analysis_samples(7.0, 30.0), 210);
        assert_eq!(expected_motion_analysis_samples(20.0, 24.0), 480);
        assert_eq!(motion_analysis_fps(60.0), 30.0);
        assert_eq!(motion_analysis_fps(24.0), 24.0);
    }

    #[test]
    fn cropped_glow_region_matches_the_full_frame_glow() {
        let points = [(60.0, 50.0), (90.0, 40.0), (130.0, 70.0)];
        let full =
            render_glow_layer(200, 120, &points, 4.0, 10.0, [255, 0, 0]).expect("glow layer");
        let (region, left, top) =
            render_glow_region(200, 120, &points, 4.0, 10.0, [255, 0, 0]).expect("glow region");
        assert!(region.width() < 200 && region.height() < 120);
        let mut composed = RgbaImage::new(200, 120);
        image::imageops::overlay(&mut composed, &region, left, top);
        let alpha = |image: &RgbaImage| image.pixels().map(|pixel| pixel.0[3]).collect::<Vec<_>>();
        assert_eq!(alpha(&composed), alpha(&full));
    }

    #[test]
    fn camera_motion_offsets_the_tracer_and_can_be_disabled() {
        let mut project = Project {
            video_path: PathBuf::new(),
            duration: 2.0,
            frame_rate: 30.0,
            trim_in: 0.0,
            trim_out: 2.0,
            keyframes: vec![Keyframe {
                time: 0.0,
                x: 0.3,
                y: 0.4,
            }],
            camera_motion: vec![
                CameraMotionSample {
                    time: 0.0,
                    x: 0.0,
                    y: 0.0,
                },
                CameraMotionSample {
                    time: 1.0,
                    x: 0.1,
                    y: -0.05,
                },
            ],
            camera_motion_enabled: true,
            masks: Vec::new(),
            ball_colour: None,
            path_smoothing: 0.5,
            trail_duration: 5.0,
            trail_pace: 0.5,
            tracer: TracerStyle::default(),
        };
        let points = project_trail_points(&project, 0.5);
        assert!((points[0].0 - 0.35).abs() < 0.0001);
        assert!((points[0].1 - 0.375).abs() < 0.0001);

        project.camera_motion_enabled = false;
        assert_eq!(camera_motion_offset(&project, 0.5), (0.0, 0.0));
        assert_eq!(project_trail_points(&project, 0.5), vec![(0.3, 0.4)]);
    }

    #[test]
    fn keyframe_editing_targets_the_keyframe_at_the_current_frame() {
        let keys = [
            Keyframe {
                time: 0.0,
                x: 0.2,
                y: 0.3,
            },
            Keyframe {
                time: 1.0,
                x: 0.7,
                y: 0.8,
            },
        ];
        assert_eq!(keyframe_index_at_time(&keys, 1.0, 30.0), Some(1));
        assert_eq!(keyframe_index_at_time(&keys, 1.02, 30.0), None);
        assert_eq!(keyframe_index_at_time(&keys, 0.01, 30.0), Some(0));
    }

    #[test]
    fn arrow_frame_steps_navigate_adjacent_frames() {
        assert_eq!(step_frame(30, 1, 0, 89), Some(31));
        assert_eq!(step_frame(31, 1, 0, 89), Some(32));
        assert_eq!(step_frame(32, -1, 0, 89), Some(31));
        assert_eq!(step_frame(89, 1, 0, 89), None);
        assert_eq!(step_frame(0, -1, 0, 89), None);
    }

    #[test]
    fn nudging_creates_keyframe_at_the_current_interpolated_position() {
        let mut project = Project {
            video_path: PathBuf::new(),
            duration: 4.0,
            frame_rate: 30.0,
            trim_in: 0.0,
            trim_out: 4.0,
            keyframes: vec![
                Keyframe {
                    time: 0.0,
                    x: 0.2,
                    y: 0.3,
                },
                Keyframe {
                    time: 2.0,
                    x: 0.8,
                    y: 0.7,
                },
                Keyframe {
                    time: 4.0,
                    x: 0.4,
                    y: 0.9,
                },
            ],
            camera_motion: Vec::new(),
            camera_motion_enabled: false,
            masks: Vec::new(),
            ball_colour: None,
            path_smoothing: 0.8,
            trail_duration: 5.0,
            trail_pace: 0.5,
            tracer: TracerStyle::default(),
        };
        let frame = step_frame(30, 1, 0, 119).and_then(|frame| step_frame(frame, 1, 0, 119));
        assert_eq!(frame, Some(32));
        let time = frame_time(frame.unwrap(), 30.0);
        let smoothed = smooth_keyframes(&project.keyframes, project.path_smoothing);
        let expected_position = position_on_path(&smoothed, time);
        let earlier_keyframe = project.keyframes[0].x;
        let index = ensure_keyframe_at_time(&mut project, time)
            .expect("nudge should create a keyframe on the current frame");
        assert_eq!(project.keyframes.len(), 4);
        assert_eq!(project.keyframes[index].time, time);
        assert!((project.keyframes[index].x - expected_position.0).abs() < 0.0001);
        assert!((project.keyframes[index].y - expected_position.1).abs() < 0.0001);
        assert!((project.keyframes[index].x - earlier_keyframe).abs() > 0.1);
    }

    #[test]
    fn nudging_before_first_keyframe_uses_the_visible_path_position() {
        let mut project = Project {
            video_path: PathBuf::new(),
            duration: 3.0,
            frame_rate: 30.0,
            trim_in: 0.0,
            trim_out: 3.0,
            keyframes: vec![Keyframe {
                time: 1.0,
                x: 0.42,
                y: 0.63,
            }],
            camera_motion: Vec::new(),
            camera_motion_enabled: false,
            masks: Vec::new(),
            ball_colour: None,
            path_smoothing: 0.5,
            trail_duration: 5.0,
            trail_pace: 0.5,
            tracer: TracerStyle::default(),
        };
        let index = ensure_keyframe_at_time(&mut project, frame_time(29, 30.0))
            .expect("nudge before first point should still seed from existing position");
        assert_eq!(project.keyframes.len(), 2);
        assert_eq!(
            (project.keyframes[index].x, project.keyframes[index].y),
            (0.42, 0.63)
        );
    }

    #[test]
    fn older_projects_default_to_motion_compensation_disabled() {
        let project: Project = serde_json::from_str(
            r#"{
                "video_path": "practice.mp4",
                "duration": 5.0,
                "frame_rate": 30.0,
                "trim_in": 0.0,
                "trim_out": 5.0,
                "keyframes": [],
                "path_smoothing": 0.5,
                "tracer": {"color": [255, 76, 48], "width": 5.0, "glow": 10.0}
            }"#,
        )
        .expect("legacy project should remain loadable");
        assert!(!project.camera_motion_enabled);
        assert!(project.camera_motion.is_empty());
        assert_eq!(project.trail_duration, default_trail_duration());
        assert_eq!(project.trail_pace, default_trail_pace());
        assert_eq!(project.tracer.glow_color, project.tracer.color);
    }

    #[test]
    fn path_smoothing_reduces_keyframe_zigzags_and_preserves_endpoints() {
        let keys = [
            Keyframe {
                time: 0.0,
                x: 0.1,
                y: 0.5,
            },
            Keyframe {
                time: 1.0,
                x: 0.3,
                y: 0.3,
            },
            Keyframe {
                time: 2.0,
                x: 0.5,
                y: 0.7,
            },
            Keyframe {
                time: 3.0,
                x: 0.7,
                y: 0.3,
            },
            Keyframe {
                time: 4.0,
                x: 0.9,
                y: 0.5,
            },
        ];
        let smoothed = smooth_keyframes(&keys, 1.0);
        assert_eq!(smoothed.first().unwrap().x, keys[0].x);
        assert_eq!(smoothed.first().unwrap().y, keys[0].y);
        assert_eq!(smoothed.last().unwrap().x, keys[4].x);
        assert_eq!(smoothed.last().unwrap().y, keys[4].y);
        assert!(smoothed[2].y < keys[2].y);
        assert!(smoothed[2].y > 0.3);
    }

    #[test]
    fn frame_navigation_uses_source_frame_rate() {
        let time = frame_time(42, 120.0);
        assert!((time - 0.35).abs() < f32::EPSILON);
        assert_eq!(time_to_frame(time, 120.0), 42);
    }

    #[test]
    fn frame_cache_evicts_the_frames_farthest_from_the_playhead() {
        let mut cache = FrameCache::new(30);
        cache.focus = 5;
        for frame in 0..3 {
            assert!(cache.insert(frame, vec![0; 10]));
        }
        assert!(cache.insert(5, vec![0; 10]));
        assert!(cache.get(0).is_none());
        assert!(cache.get(1).is_some() && cache.get(2).is_some() && cache.get(5).is_some());
        assert!(!cache.insert(100, vec![0; 10]));
        assert!(cache.get(100).is_none());
        assert_eq!(cache.bytes, 30);
    }

    #[test]
    #[ignore = "needs FFmpeg and OPENTRACER_TEST_VIDEO pointing at a video"]
    fn preview_worker_serves_seeks_and_playback_from_the_frame_cache() {
        let Ok(video) = std::env::var("OPENTRACER_TEST_VIDEO") else {
            return;
        };
        let (command_sender, commands) = std::sync::mpsc::channel();
        let (results, receiver) = std::sync::mpsc::sync_channel(2);
        std::thread::spawn(move || {
            super::preview_worker(commands, results, eframe::egui::Context::default())
        });
        let request = |id: u64, frame: u64| super::PreviewRequest {
            id,
            path: PathBuf::from(&video),
            time: frame as f32 / 30.0,
            frame_rate: 30.0,
            frame_range: (0, 150),
        };
        let wait = |id: u64| loop {
            let result = receiver
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("preview result");
            if result.id == id {
                return result;
            }
        };

        let started = Instant::now();
        command_sender
            .send(super::PreviewCommand::Seek(request(1, 60)))
            .unwrap();
        let cold = wait(1);
        println!("cold seek: {:?}", started.elapsed());
        let frame = cold.result.expect("cold frame");
        println!("preview size: {:?}", frame.size);

        for (id, target) in [(2, 61), (3, 62)] {
            let started = Instant::now();
            command_sender
                .send(super::PreviewCommand::Seek(request(id, target)))
                .unwrap();
            wait(id).result.expect("step frame");
            println!("step to {target}: {:?}", started.elapsed());
        }

        // Let the background decode fill the trimmed range, then jump backwards.
        std::thread::sleep(std::time::Duration::from_secs(40));
        let started = Instant::now();
        command_sender
            .send(super::PreviewCommand::Seek(request(4, 5)))
            .unwrap();
        wait(4).result.expect("cached backwards seek");
        println!("cached backwards seek: {:?}", started.elapsed());

        let started = Instant::now();
        command_sender
            .send(super::PreviewCommand::Play(request(5, 0)))
            .unwrap();
        let mut frames = 0;
        loop {
            let result = wait(5);
            if result.finished {
                break;
            }
            frames += 1;
        }
        println!("played {frames} frames in {:?}", started.elapsed());
        assert!(frames >= 140, "only {frames} frames were delivered");
    }

    #[test]
    fn concave_mask_shapes_triangulate_to_their_exact_area() {
        let l_shape = [
            (0.0, 0.0),
            (0.6, 0.0),
            (0.6, 0.2),
            (0.2, 0.2),
            (0.2, 0.6),
            (0.0, 0.6),
        ];
        let triangles = super::triangulate_polygon(&l_shape);
        assert_eq!(triangles.len(), 4);
        let area: f32 = triangles
            .iter()
            .map(|t| super::polygon_area(&[l_shape[t[0]], l_shape[t[1]], l_shape[t[2]]]).abs())
            .sum();
        assert!((area - super::polygon_area(&l_shape).abs()).abs() < 1e-5);
        let reversed: Vec<_> = l_shape.iter().rev().copied().collect();
        assert_eq!(super::triangulate_polygon(&reversed).len(), 4);
    }

    #[test]
    fn export_mask_hides_the_tracer_only_inside_the_shape() {
        let masks = [super::MaskRegion::new(vec![
            (0.4, 0.0),
            (0.6, 0.0),
            (0.6, 1.0),
            (0.4, 1.0),
        ])];
        let coverage = super::mask_coverage_image(100, 10, &masks, 0.0).expect("mask coverage");
        let mut image = RgbaImage::from_pixel(100, 10, Rgba([255, 0, 0, 255]));
        super::apply_mask_coverage(&mut image, &coverage);
        assert_eq!(image.get_pixel(10, 5).0[3], 255);
        assert_eq!(image.get_pixel(90, 5).0[3], 255);
        assert_eq!(image.get_pixel(50, 5).0[3], 0);
        assert!(super::mask_coverage_image(100, 10, &[], 0.0).is_none());
    }

    fn square(x: f32) -> Vec<(f32, f32)> {
        vec![(x, 0.0), (x + 0.2, 0.0), (x + 0.2, 0.2), (x, 0.2)]
    }

    #[test]
    fn animated_mask_interpolates_between_keyframes_and_holds_at_the_ends() {
        let mut mask = super::MaskRegion::new(square(0.0));
        assert_eq!(mask.shape_at(5.0), square(0.0));
        mask.ensure_keyframe(1.0, 30.0);
        mask.set_vertex(0, (0.5, 0.0), 3.0, 30.0);
        assert_eq!(mask.keyframes.len(), 2);
        assert_eq!(mask.shape_at(0.0), square(0.0));
        assert_eq!(mask.shape_at(9.0)[0], (0.5, 0.0));
        let halfway = mask.shape_at(2.0);
        assert!((halfway[0].0 - 0.25).abs() < 1e-5);
        assert_eq!(halfway[1], (0.2, 0.0));
        // Editing at an existing keyframe time updates it instead of adding another.
        mask.set_vertex(0, (0.6, 0.0), 3.0, 30.0);
        assert_eq!(mask.keyframes.len(), 2);
    }

    #[test]
    fn mask_vertices_can_be_added_and_removed_across_all_keyframes() {
        let mut mask = super::MaskRegion::new(square(0.0));
        mask.ensure_keyframe(0.0, 30.0);
        mask.ensure_keyframe(2.0, 30.0);
        mask.insert_vertex(0, (0.1, 0.05), 1.0, 30.0);
        assert_eq!(mask.points.len(), 5);
        assert_eq!(mask.keyframes.len(), 3);
        assert!(mask.keyframes.iter().all(|k| k.points.len() == 5));
        assert_eq!(mask.shape_at(1.0)[1], (0.1, 0.05));
        mask.remove_vertex(1);
        assert!(mask.keyframes.iter().all(|k| k.points.len() == 4));
        mask.remove_vertex(1);
        mask.remove_vertex(1);
        assert_eq!(mask.points.len(), 3, "a mask always keeps three corners");
    }

    #[test]
    fn export_mask_follows_keyframes_over_time() {
        let mut mask = super::MaskRegion::new(vec![(0.0, 0.0), (0.2, 0.0), (0.2, 1.0), (0.0, 1.0)]);
        mask.ensure_keyframe(0.0, 30.0);
        for vertex in 0..4 {
            let (x, y) = mask.points[vertex];
            mask.set_vertex(vertex, (x + 0.6, y), 1.0, 30.0);
        }
        let masks = [mask];
        let early = super::mask_coverage_image(100, 10, &masks, 0.0).unwrap();
        let late = super::mask_coverage_image(100, 10, &masks, 1.0).unwrap();
        assert!(early.get_pixel(10, 5).0[0] > 200 && late.get_pixel(10, 5).0[0] == 0);
        assert!(late.get_pixel(70, 5).0[0] > 200 && early.get_pixel(70, 5).0[0] == 0);
    }

    #[test]
    fn projects_saved_before_masks_load_without_any() {
        let project: Project = serde_json::from_str(
            r#"{
                "video_path": "practice.mp4",
                "duration": 5.0,
                "trim_in": 0.0,
                "trim_out": 5.0,
                "keyframes": [],
                "tracer": {"color": [255, 76, 48], "width": 5.0, "glow": 10.0}
            }"#,
        )
        .expect("legacy project should remain loadable");
        assert!(project.masks.is_empty());
    }

    #[test]
    fn uncached_span_skips_frames_already_cached_at_both_ends() {
        let mut cache = FrameCache::new(1000);
        for frame in [10, 11, 19, 20] {
            cache.insert(frame, vec![0; 1]);
        }
        assert_eq!(cache.uncached_span(10, 20), Some((12, 18)));
        assert_eq!(cache.uncached_span(10, 11), None);
        assert_eq!(cache.uncached_span(30, 31), Some((30, 31)));
    }

    #[test]
    fn trim_positions_snap_to_frames_but_keep_video_end() {
        assert!((snap_trim_time(1.016, 10.0, 30.0) - 1.0).abs() < 0.0001);
        assert_eq!(snap_trim_time(9.99, 10.0, 30.0), 10.0);
    }

    #[test]
    fn tracer_stroke_is_continuous_with_a_feathered_edge() {
        let mut image = RgbaImage::new(100, 40);
        draw_feathered_path(
            &mut image,
            &[(5.0, 20.0), (95.0, 20.0)],
            4.0,
            6.0,
            [40, 220, 80],
            [40, 220, 80],
        );
        assert_eq!(image.get_pixel(50, 20).0[3], 255);
        let edge_alpha: Vec<u8> = (24..=35).map(|y| image.get_pixel(50, y).0[3]).collect();
        assert!(edge_alpha[0] > 0 && edge_alpha[0] < 255);
        assert!(edge_alpha.windows(2).all(|pair| pair[0] >= pair[1]));
        for x in 10..90 {
            assert_eq!(image.get_pixel(x, 20), &Rgba([40, 220, 80, 255]));
        }
    }

    #[test]
    fn later_curve_glow_does_not_cover_earlier_solid_tracer_core() {
        let mut image = RgbaImage::new(100, 40);
        draw_feathered_path(
            &mut image,
            &[(5.0, 20.0), (50.0, 20.0), (5.0, 22.0)],
            4.0,
            8.0,
            [40, 220, 80],
            [40, 220, 80],
        );
        assert_eq!(image.get_pixel(30, 20), &Rgba([40, 220, 80, 255]));
    }

    #[test]
    fn glow_can_use_a_different_color_from_the_tracer_core() {
        let line_color = [40, 220, 80];
        let glow_color = [240, 20, 20];
        let mut image = RgbaImage::new(100, 40);
        draw_feathered_path(
            &mut image,
            &[(5.0, 20.0), (95.0, 20.0)],
            4.0,
            6.0,
            line_color,
            glow_color,
        );
        assert_eq!(image.get_pixel(50, 20), &Rgba([40, 220, 80, 255]));
        assert!((0..40).any(|y| {
            let pixel = image.get_pixel(50, y);
            pixel.0[0] == glow_color[0] && pixel.0[1] == glow_color[1] && pixel.0[3] > 0
        }));
    }

    #[test]
    fn mjpeg_reader_uses_complete_markers_and_preserves_following_frames() {
        let bytes = [
            0xFF, 0xD8, 0x01, 0xD9, 0x02, 0xFF, 0xD9, 0xFF, 0xD8, 0x03, 0xFF, 0xD9,
        ];
        let mut reader = Cursor::new(bytes);
        let mut buffered = Vec::new();
        assert_eq!(
            read_mjpeg_frame(&mut reader, &mut buffered).unwrap(),
            Some(vec![0xFF, 0xD8, 0x01, 0xD9, 0x02, 0xFF, 0xD9])
        );
        assert_eq!(
            read_mjpeg_frame(&mut reader, &mut buffered).unwrap(),
            Some(vec![0xFF, 0xD8, 0x03, 0xFF, 0xD9])
        );
        assert_eq!(read_mjpeg_frame(&mut reader, &mut buffered).unwrap(), None);
    }

    #[test]
    fn predicted_ball_arc_passes_through_anchors_and_peaks_between_them() {
        let anchors = [(0.0, 0.0, 100.0), (1.0, 50.0, 0.0), (2.0, 100.0, 100.0)];
        for anchor in anchors {
            let (x, y) = super::predict_ball_position(&anchors, anchor.0);
            assert!((x - anchor.1).abs() < 1e-3 && (y - anchor.2).abs() < 1e-3);
        }
        let (x, y) = super::predict_ball_position(&anchors, 0.5);
        assert!((x - 25.0).abs() < 1e-3);
        assert!((y - 25.0).abs() < 1e-3, "parabola through start, peak, end");
        assert_eq!(super::predict_ball_position(&anchors, 9.0), (100.0, 100.0));
    }

    fn frame_with_ball(width: usize, height: usize, ball: Option<(usize, usize)>) -> Vec<u8> {
        let mut frame = vec![60_u8; width * height];
        if let Some((bx, by)) = ball {
            for y in by - 1..=by + 1 {
                for x in bx - 1..=bx + 1 {
                    frame[y * width + x] = 250;
                }
            }
        }
        frame
    }

    #[test]
    fn find_ball_picks_the_moving_blob_near_the_prediction() {
        let (width, height) = (200, 120);
        let previous = frame_with_ball(width, height, Some((60, 50)));
        let mut current = frame_with_ball(width, height, Some((70, 48)));
        let next = frame_with_ball(width, height, Some((80, 46)));
        // A static bright patch nearer the prediction must lose to the moving ball.
        for y in 59..=61 {
            for x in 68..=70 {
                current[y * width + x] = 250;
            }
        }
        let mut previous = previous;
        let mut next = next;
        for frame in [&mut previous, &mut next] {
            for y in 59..=61 {
                for x in 68..=70 {
                    frame[y * width + x] = 250;
                }
            }
        }
        let (x, y, _) = super::find_ball(
            Some(&previous),
            &current,
            Some(&next),
            width,
            height,
            (72.0, 52.0),
            15.0,
        )
        .expect("ball found");
        assert!(
            (x - 70.0).abs() <= 1.0 && (y - 48.0).abs() <= 1.0,
            "{x},{y}"
        );
        let empty = frame_with_ball(width, height, None);
        assert!(super::find_ball(None, &empty, None, width, height, (72.0, 52.0), 15.0).is_none());
    }

    #[test]
    fn follow_ball_tracks_a_synthetic_flight_that_deviates_from_the_arc() {
        let (width, height) = (200_usize, 120_usize);
        let frame_rate = 10.0;
        // The real ball drifts 8 px below the straight-line guess between the two anchors.
        let truth = |index: usize| {
            let progress = index as f32 / 20.0;
            (
                20.0 + 160.0 * progress,
                60.0 + 8.0 * (progress * std::f32::consts::PI).sin(),
            )
        };
        let mut index = 0_usize;
        let frames = || {
            if index > 20 {
                return Ok(None);
            }
            let (x, y) = truth(index);
            index += 1;
            let gray = frame_with_ball(
                width,
                height,
                Some((x.round() as usize, y.round() as usize)),
            );
            Ok(Some(gray.iter().flat_map(|value| [*value; 3]).collect()))
        };
        let anchors = [
            (0.0, truth(0).0, truth(0).1),
            (2.0, truth(20).0, truth(20).1),
        ];
        let positions = super::follow_ball(
            frames,
            &anchors,
            super::TrackSettings {
                width,
                height,
                frame_rate,
                picked_colour: None,
            },
            |_, _| {},
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(positions.len(), 21);
        for (index, (_, x, y)) in positions.iter().enumerate() {
            let (tx, ty) = truth(index);
            assert!(
                (x - tx).abs() <= 1.5 && (y - ty).abs() <= 1.5,
                "frame {index}"
            );
        }
    }

    fn rgb_frame(width: usize, height: usize, patches: &[(usize, usize, [u8; 3])]) -> Vec<u8> {
        let mut frame: Vec<u8> = (0..width * height).flat_map(|_| [70_u8, 110, 70]).collect();
        for (px, py, colour) in patches {
            for y in py - 2..=py + 2 {
                for x in px - 2..=px + 2 {
                    frame[(y * width + x) * 3..(y * width + x) * 3 + 3].copy_from_slice(colour);
                }
            }
        }
        frame
    }

    #[test]
    fn picking_a_colour_finds_saturated_balls_and_rejects_white() {
        let (width, height) = (40_usize, 30_usize);
        let mut rgba = vec![255_u8; width * height * 4];
        for y in 10..=14 {
            for x in 20..=24 {
                let index = (y * width + x) * 4;
                rgba[index..index + 3].copy_from_slice(&[250, 220, 20]);
            }
        }
        let yellow = super::pick_ball_colour(&rgba, [width, height], (22.0 / 40.0, 12.0 / 30.0));
        assert_eq!(yellow, Some([250, 220, 20]));
        assert_eq!(
            super::pick_ball_colour(&rgba, [width, height], (5.0 / 40.0, 25.0 / 30.0)),
            None
        );
    }

    #[test]
    fn follow_ball_uses_the_picked_colour_even_when_the_arc_is_far_off() {
        let (width, height) = (300_usize, 160_usize);
        let yellow = [250_u8, 220, 20];
        // Only two anchors (a straight guess), but the ball really rises 40 px above that line,
        // with a brighter white patch placed right on the guess to distract a brightness tracker.
        let truth = |index: usize| {
            let progress = index as f32 / 20.0;
            (
                30.0 + 240.0 * progress,
                100.0 - 50.0 * (progress * std::f32::consts::PI).sin(),
            )
        };
        let mut index = 0_usize;
        let frames = || {
            if index > 20 {
                return Ok(None);
            }
            let (x, y) = truth(index);
            let (gx, gy) = (30.0 + 240.0 * index as f32 / 20.0, 100.0);
            index += 1;
            Ok(Some(rgb_frame(
                width,
                height,
                &[
                    (gx.round() as usize, gy as usize, [255, 255, 255]),
                    (x.round() as usize, y.round() as usize, yellow),
                ],
            )))
        };
        let anchors = [
            (0.0, truth(0).0, truth(0).1),
            (2.0, truth(20).0, truth(20).1),
        ];
        let positions = super::follow_ball(
            frames,
            &anchors,
            super::TrackSettings {
                width,
                height,
                frame_rate: 10.0,
                picked_colour: Some(super::rgb_chroma(yellow)),
            },
            |_, _| {},
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
        for (index, (_, x, y)) in positions.iter().enumerate() {
            let (tx, ty) = truth(index);
            assert!(
                (x - tx).abs() <= 2.0 && (y - ty).abs() <= 2.0,
                "frame {index}: {x},{y}"
            );
        }
    }

    #[test]
    fn whole_mask_moves_and_stays_inside_the_frame() {
        let mut mask = super::MaskRegion::new(vec![(0.1, 0.1), (0.3, 0.1), (0.3, 0.3)]);
        mask.translate((0.2, 0.1), 0.0, 30.0);
        assert!((mask.points[0].0 - 0.3).abs() < 1e-5 && (mask.points[2].1 - 0.4).abs() < 1e-5);
        mask.translate((5.0, -5.0), 0.0, 30.0);
        assert!((mask.points[1].0 - 1.0).abs() < 1e-5 && mask.points[0].1.abs() < 1e-5);
        let mut animated = super::MaskRegion::new(vec![(0.1, 0.1), (0.3, 0.1), (0.3, 0.3)]);
        animated.ensure_keyframe(0.0, 30.0);
        animated.translate((0.1, 0.0), 1.0, 30.0);
        assert_eq!(animated.keyframes.len(), 2);
        assert!((animated.shape_at(1.0)[0].0 - 0.2).abs() < 1e-5);
        assert!((animated.shape_at(0.0)[0].0 - 0.1).abs() < 1e-5);
    }

    #[test]
    fn point_in_polygon_handles_concave_shapes() {
        use crate::egui::Pos2;
        let shape = [
            Pos2::new(0.0, 0.0),
            Pos2::new(10.0, 0.0),
            Pos2::new(10.0, 10.0),
            Pos2::new(5.0, 4.0),
            Pos2::new(0.0, 10.0),
        ];
        assert!(super::point_in_polygon(Pos2::new(2.0, 2.0), &shape));
        assert!(!super::point_in_polygon(Pos2::new(5.0, 8.0), &shape));
        assert!(!super::point_in_polygon(Pos2::new(12.0, 2.0), &shape));
    }
}
