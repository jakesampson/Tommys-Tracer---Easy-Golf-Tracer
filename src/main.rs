use std::{
    fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use image::{GrayImage, ImageFormat, Luma, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use imageproc::filter::box_filter;
use serde::{Deserialize, Serialize};

const EXPORT_FPS: u32 = 30;
const FRAME_LIMIT: u32 = 60 * 60 * 30;
const PREVIEW_MAX_DIMENSION: u32 = 720;
const TRACER_REFERENCE_HEIGHT: f32 = 1080.0;
const TRACER_WIDTH_MAX: f32 = 16.0;
const TRACER_GLOW_MAX: f32 = 24.0;
const STYLE_SCALE_MAX: i32 = 10;
const MOTION_ANALYSIS_FPS: u32 = 10;

fn expected_motion_analysis_samples(duration: f32) -> usize {
    (duration * MOTION_ANALYSIS_FPS as f32).ceil().max(1.0) as usize
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Keyframe {
    time: f32,
    x: f32,
    y: f32,
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
    #[serde(default = "default_path_smoothing")]
    path_smoothing: f32,
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

fn style_scale_to_internal(value: i32, internal_max: f32) -> f32 {
    value.clamp(1, STYLE_SCALE_MAX) as f32 / STYLE_SCALE_MAX as f32 * internal_max
}

fn internal_to_style_scale(value: f32, internal_max: f32) -> i32 {
    ((value / internal_max).clamp(0.0, 1.0) * STYLE_SCALE_MAX as f32)
        .round()
        .clamp(1.0, STYLE_SCALE_MAX as f32) as i32
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TracerStyle {
    color: [u8; 3],
    width: f32,
    glow: f32,
}

impl Default for TracerStyle {
    fn default() -> Self {
        Self {
            color: [255, 76, 48],
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
    tracer_glow: Option<egui::TextureHandle>,
    preview_size: [usize; 2],
    motion_receiver: Option<Receiver<MotionAnalysisMessage>>,
    motion_cancel: Option<Arc<AtomicBool>>,
    motion_analysis_selection: Option<(PathBuf, f32, f32)>,
    motion_analysis_frames: usize,
    motion_analysis_total: usize,
    motion_analysis_progress: f32,
    export_receiver: Option<Receiver<ExportMessage>>,
    export_progress: f32,
    export_phase: String,
    status: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppScreen {
    Welcome,
    Trim,
    Stabilize,
    Trace,
}

struct PreviewRequest {
    id: u64,
    path: PathBuf,
    time: f32,
    frame_rate: f32,
}

enum PreviewCommand {
    Seek(PreviewRequest),
    Play {
        id: u64,
        path: PathBuf,
        time: f32,
        frame_rate: f32,
    },
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

struct PlaybackStream {
    child: Child,
    frames: Receiver<Result<PreviewFrame, String>>,
    path: PathBuf,
    id: u64,
    time: f32,
    frame_rate: f32,
}

enum ExportMessage {
    Progress { fraction: f32, phase: String },
    Finished(Result<PathBuf, String>),
}

enum MotionAnalysisMessage {
    Progress { completed: usize, total: usize },
    Finished(Result<Vec<CameraMotionSample>, String>),
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
            tracer_glow: None,
            preview_size: [0, 0],
            motion_receiver: None,
            motion_cancel: None,
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
                    path_smoothing: default_path_smoothing(),
                    tracer: TracerStyle::default(),
                });
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
        if let Err(error) = self.preview_sender.send(PreviewCommand::Play {
            id: self.preview_request_id,
            path: project.video_path.clone(),
            time: frame_time(self.playhead_frame, project.frame_rate),
            frame_rate: project.frame_rate,
        }) {
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
            .add_filter("OpenTracer project", &["otracer"])
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
        let expected_frames = expected_motion_analysis_samples(project.trim_out - project.trim_in);
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
                self.status = "Analyzing camera movement...".to_owned();
            }
            Err(error) => {
                self.motion_analysis_selection = None;
                self.status = format!("Could not start camera-motion analysis: {error}");
            }
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
            .add_filter("OpenTracer project", &["otracer", "json"])
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
        egui::CentralPanel::default().show(ctx, |ui| {
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
                            ui.label(
                                egui::RichText::new("◉")
                                    .size(48.0)
                                    .color(Color32::from_rgb(89, 177, 255)),
                            );
                            ui.heading(egui::RichText::new("OpenTracer").size(36.0).strong());
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new("Welcome to OpenTracer")
                                    .size(19.0)
                                    .color(Color32::from_rgb(192, 211, 232)),
                            );
                            ui.add_space(8.0);
                            ui.label("Trace the ball flight in your golf videos, frame by frame.");
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
                                    egui::Button::new("Open a saved project")
                                        .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 16)),
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
                        if ui.button("Continue to stabilization").clicked() {
                            if self.is_playing {
                                self.pause_playback();
                            }
                            self.screen = AppScreen::Stabilize;
                        }
                    });
                });
            });

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
                    (&self.preview, self.preview_size, &mut self.tracer_glow),
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

    fn show_stabilization_screen(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("stabilization_toolbar")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(21, 32, 45, 245))
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Back to trim").clicked() {
                        self.cancel_motion_analysis();
                        self.screen = AppScreen::Trim;
                    }
                    ui.separator();
                    ui.label("2. Stabilize camera motion (experimental)");
                });
            });

        let Some(project) = &mut self.project else {
            self.screen = AppScreen::Welcome;
            return;
        };
        let mut start_analysis = false;
        let mut cancel_analysis = false;
        let mut continue_to_trace = None;
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(30.0);
                ui.heading("Keep the tracer anchored to the scene");
                ui.add_space(10.0);
                ui.label(
                    "Analyze the trimmed clip for small camera translations. \
                     Motion compensation moves the tracer with the scene; \
                     it does not crop or alter the source video.",
                );
                ui.add_space(20.0);
                if self.motion_receiver.is_some() {
                    ui.add(
                        egui::ProgressBar::new(self.motion_analysis_progress)
                            .desired_width(420.0)
                            .text(format!(
                                "Analyzed {} / {} frames ({:.0}%)",
                                self.motion_analysis_frames,
                                self.motion_analysis_total,
                                self.motion_analysis_progress * 100.0
                            )),
                    );
                    if ui.button("Cancel analysis").clicked() {
                        cancel_analysis = true;
                    }
                } else if ui.button("Analyze camera motion").clicked() {
                    start_analysis = true;
                }
                if !project.camera_motion.is_empty() {
                    ui.add_space(14.0);
                    ui.checkbox(
                        &mut project.camera_motion_enabled,
                        "Apply motion compensation",
                    );
                    ui.label(format!(
                        "Analyzed {} motion samples. Turn compensation off at any time to compare.",
                        project.camera_motion.len()
                    ));
                } else {
                    ui.add_space(14.0);
                    ui.label("You can skip this experiment and use the original footage.");
                }
                ui.add_space(24.0);
                if ui.button("Continue to tracing").clicked() {
                    continue_to_trace = Some(false);
                }
                if ui.button("Skip stabilization and continue").clicked() {
                    continue_to_trace = Some(true);
                }
            });
        });
        if start_analysis {
            self.analyze_camera_motion(ctx);
        }
        if cancel_analysis {
            self.cancel_motion_analysis();
            self.status = "Camera-motion analysis cancelled.".to_owned();
        }
        if let Some(skip_stabilization) = continue_to_trace {
            self.cancel_motion_analysis();
            if skip_stabilization {
                if let Some(project) = &mut self.project {
                    project.camera_motion_enabled = false;
                }
            }
            if self.is_playing {
                self.pause_playback();
            }
            self.screen = AppScreen::Trace;
            if let Some(project) = &self.project {
                self.playhead_frame = time_to_frame(project.trim_in, project.frame_rate);
            }
            self.request_preview();
        }
    }

    fn show_trace_screen(&mut self, ctx: &egui::Context) {
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
                    ui.label("3. Trace ball flight");
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
                        if !project.camera_motion.is_empty() {
                            ui.checkbox(&mut project.camera_motion_enabled, "Motion compensation");
                        }
                    }
                });
            });

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
                    let mut color = Color32::from_rgb(
                        project.tracer.color[0],
                        project.tracer.color[1],
                        project.tracer.color[2],
                    );
                    if ui.color_edit_button_srgba(&mut color).changed() {
                        project.tracer.color = [color.r(), color.g(), color.b()];
                    }
                    let mut width_scale =
                        internal_to_style_scale(project.tracer.width, TRACER_WIDTH_MAX);
                    if ui
                        .add(egui::Slider::new(&mut width_scale, 1..=STYLE_SCALE_MAX).text("Width"))
                        .on_hover_text("Style scale: 1–10, where 10 is 100%.")
                        .changed()
                    {
                        project.tracer.width =
                            style_scale_to_internal(width_scale, TRACER_WIDTH_MAX);
                    }
                    let mut glow_scale =
                        internal_to_style_scale(project.tracer.glow, TRACER_GLOW_MAX);
                    if ui
                        .add(egui::Slider::new(&mut glow_scale, 1..=STYLE_SCALE_MAX).text("Glow"))
                        .on_hover_text("Style scale: 1–10, where 10 is 100%.")
                        .changed()
                    {
                        project.tracer.glow = style_scale_to_internal(glow_scale, TRACER_GLOW_MAX);
                    }
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
                    ui.separator();
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.heading("Keyframes");
                        if ui
                            .small_button(
                                egui::RichText::new("Clear")
                                    .color(Color32::from_rgb(255, 151, 151)),
                            )
                            .clicked()
                        {
                            project.keyframes.clear();
                        }
                    });
                    if project.keyframes.is_empty() {
                        ui.label("Click the video preview to place a keyframe.");
                    }
                    let mut remove_index = None;
                    for (index, keyframe) in project.keyframes.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "{:.3}s  ({:.2}, {:.2})",
                                keyframe.time, keyframe.x, keyframe.y
                            ));
                            if ui.small_button("x").clicked() {
                                remove_index = Some(index);
                            }
                        });
                    }
                    if let Some(index) = remove_index {
                        project.keyframes.remove(index);
                    }
                }
            });

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
                (&self.preview, self.preview_size, &mut self.tracer_glow),
                false,
            );
        });
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
                    self.status = format!("Preview: frame {} ({time:.3}s)", self.playhead_frame);
                }
                Err(error) => self.status = error,
            }
        }

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
                        self.status = "Camera-motion analysis complete.".to_owned();
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

        if !ctx.wants_keyboard_input()
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

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
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
            AppScreen::Stabilize => self.show_stabilization_screen(ctx),
            AppScreen::Trace => self.show_trace_screen(ctx),
        }
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
    if let Some(texture) = preview_data.0 {
        let aspect = preview_data.1[0] as f32 / preview_data.1[1].max(1) as f32;
        let size = fit_size(video_area.size(), aspect);
        let rect = Rect::from_center_size(video_area.center(), size);
        let response = ui.interact(rect, ui.id().with("video-preview"), Sense::click());
        ui.painter().image(
            texture.id(),
            rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        if !trim_mode {
            draw_preview_tracer(
                ui,
                rect,
                project,
                frame_time(*playhead_frame, project.frame_rate),
                preview_data.2,
            );
            if response.clicked() {
                if let Some(position) = response.interact_pointer_pos() {
                    let time = frame_time(*playhead_frame, project.frame_rate);
                    let (offset_x, offset_y) = camera_motion_offset(project, time);
                    add_keyframe(
                        project,
                        time,
                        (((position.x - rect.left()) / rect.width()) - offset_x).clamp(0.0, 1.0),
                        (((position.y - rect.top()) / rect.height()) - offset_y).clamp(0.0, 1.0),
                    );
                }
            }
        }
    } else {
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
                    if ui.button("|◀").clicked() {
                        interaction.stop_playback = *is_playing;
                        *playhead_frame = first_frame;
                        interaction.refresh_preview = true;
                    }
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
                    if ui.button("▶|").clicked() {
                        *playhead_frame = last_edit_frame;
                        interaction.stop_playback = *is_playing;
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

fn frame_time(frame: u64, frame_rate: f32) -> f32 {
    frame as f32 / frame_rate
}

fn time_to_frame(time: f32, frame_rate: f32) -> u64 {
    (time.max(0.0) * frame_rate).round() as u64
}

fn last_frame(project: &Project) -> u64 {
    (project.duration * project.frame_rate).ceil().max(1.0) as u64 - 1
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

    if response.drag_started() {
        if let Some(pointer) = response.interact_pointer_pos() {
            *drag_mode = Some(
                if allow_trimming && (pointer.x - trim_start_x).abs() <= 10.0 {
                    TimelineDrag::TrimIn
                } else if allow_trimming && (pointer.x - trim_end_x).abs() <= 10.0 {
                    TimelineDrag::TrimOut
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

fn draw_preview_tracer(
    ui: &egui::Ui,
    rect: Rect,
    project: &Project,
    time: f32,
    glow_texture: &mut Option<egui::TextureHandle>,
) {
    let (width, glow) = scaled_tracer_dimensions(&project.tracer, rect.height());
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
        project.tracer.color,
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
            ui.painter().image(
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
        draw_preview_segment(ui, segment[0], segment[1], width, color);
    }
    if let Some(point) = points.last() {
        ui.painter().circle_filled(*point, width, color);
    }
}

fn scaled_tracer_dimensions(style: &TracerStyle, canvas_height: f32) -> (f32, f32) {
    let scale = canvas_height / TRACER_REFERENCE_HEIGHT;
    (style.width * scale, style.glow * scale)
}

fn draw_preview_segment(ui: &egui::Ui, start: Pos2, end: Pos2, width: f32, color: Color32) {
    let distance = start.distance(end);
    if distance <= 0.0 {
        ui.painter()
            .line_segment([start, end], Stroke::new(width, color));
        return;
    }
    ui.painter()
        .line_segment([start, end], Stroke::new(width, color));
}

fn probe_video_info(path: &Path) -> Result<(f32, f32), String> {
    Ok((probe_duration(path)?, probe_frame_rate(path)?))
}

fn probe_frame_rate(path: &Path) -> Result<f32, String> {
    for entry in ["avg_frame_rate", "r_frame_rate"] {
        let output = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
            .arg(format!("stream={entry}"))
            .args(["-of", "default=noprint_wrappers=1:nokey=1"])
            .arg(path)
            .output()
            .map_err(|error| format!("Could not start ffprobe: {error}"))?;
        if !output.status.success() {
            continue;
        }
        if let Some(frame_rate) = String::from_utf8_lossy(&output.stdout)
            .trim()
            .split_once('/')
            .and_then(|(numerator, denominator)| {
                let numerator = numerator.parse::<f32>().ok()?;
                let denominator = denominator.parse::<f32>().ok()?;
                let frame_rate = numerator / denominator;
                (frame_rate.is_finite() && frame_rate > 0.0).then_some(frame_rate)
            })
        {
            return Ok(frame_rate);
        }
    }
    Err("Could not determine a valid video frame rate with ffprobe.".to_owned())
}

fn probe_duration(path: &Path) -> Result<f32, String> {
    let output = Command::new("ffprobe")
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
    let mut playback: Option<PlaybackStream> = None;
    let mut is_playing = false;
    loop {
        let command = if is_playing {
            match commands.recv_timeout(Duration::from_millis(2)) {
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
            match command {
                PreviewCommand::Seek(request) => {
                    is_playing = false;
                    stop_preview_process(&mut playback);
                    match start_preview_stream(
                        request.id,
                        &request.path,
                        request.time,
                        request.frame_rate,
                    ) {
                        Ok((stream, first_frame)) => {
                            let _ = results.send(PreviewResult {
                                id: request.id,
                                time: request.time,
                                result: Ok(first_frame),
                                finished: false,
                            });
                            playback = Some(stream);
                            repaint_context.request_repaint();
                        }
                        Err(error) => {
                            let _ = results.send(PreviewResult {
                                id: request.id,
                                time: request.time,
                                result: Err(error),
                                finished: false,
                            });
                            repaint_context.request_repaint();
                        }
                    }
                }
                PreviewCommand::Play {
                    id,
                    path,
                    time,
                    frame_rate,
                } => {
                    let stream_is_ready = playback.as_ref().is_some_and(|stream| {
                        preview_stream_is_ready(
                            &stream.path,
                            stream.time,
                            stream.frame_rate,
                            &path,
                            time,
                            frame_rate,
                        )
                    });
                    if stream_is_ready {
                        if let Some(stream) = &mut playback {
                            stream.id = id;
                        }
                        is_playing = true;
                    } else {
                        stop_preview_process(&mut playback);
                        match start_preview_stream(id, &path, time, frame_rate) {
                            Ok((stream, first_frame)) => {
                                let _ = results.send(PreviewResult {
                                    id,
                                    time,
                                    result: Ok(first_frame),
                                    finished: false,
                                });
                                playback = Some(stream);
                                is_playing = true;
                                repaint_context.request_repaint();
                            }
                            Err(error) => {
                                let _ = results.send(PreviewResult {
                                    id,
                                    time,
                                    result: Err(error),
                                    finished: false,
                                });
                                is_playing = false;
                                repaint_context.request_repaint();
                            }
                        }
                    }
                }
                PreviewCommand::Pause => {
                    is_playing = false;
                }
            }
            continue;
        }

        if is_playing {
            if let Some(stream) = playback.as_mut() {
                match stream.frames.try_recv() {
                    Ok(frame) => {
                        let frame_time = stream.time;
                        stream.time += 1.0 / stream.frame_rate;
                        let _ = results.try_send(PreviewResult {
                            id: stream.id,
                            time: frame_time,
                            result: frame,
                            finished: false,
                        });
                        repaint_context.request_repaint();
                        if stream.child.try_wait().ok().flatten().is_some() {
                            let _ = results.send(PreviewResult {
                                id: stream.id,
                                time: stream.time,
                                result: Ok(PreviewFrame::default()),
                                finished: true,
                            });
                            repaint_context.request_repaint();
                            playback = None;
                            is_playing = false;
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if stream.child.try_wait().ok().flatten().is_some() {
                            let _ = results.send(PreviewResult {
                                id: stream.id,
                                time: stream.time,
                                result: Ok(PreviewFrame::default()),
                                finished: true,
                            });
                            repaint_context.request_repaint();
                            playback = None;
                            is_playing = false;
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            } else {
                is_playing = false;
            }
        }
    }
    stop_preview_process(&mut playback);
}

fn start_preview_stream(
    id: u64,
    path: &Path,
    time: f32,
    frame_rate: f32,
) -> Result<(PlaybackStream, PreviewFrame), String> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-ss",
        ])
        .arg(format!("{time:.6}"))
        .arg("-i")
        .arg(path)
        .args(["-an", "-sn", "-dn", "-vf"])
        .arg(format!(
            "fps={frame_rate:.6},scale={PREVIEW_MAX_DIMENSION}:{PREVIEW_MAX_DIMENSION}:force_original_aspect_ratio=decrease"
        ))
        .args([
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-flush_packets",
            "1",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Could not start FFmpeg playback preview: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Could not read FFmpeg playback frames.".to_owned())?;
    let (sender, frames) = mpsc::sync_channel(2);
    let reader_thread = std::thread::Builder::new()
        .name("opentracer-frame-reader".to_owned())
        .spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut buffered = Vec::new();
            let started_at = Instant::now();
            let mut frame_index = 0;
            while let Ok(Some(bytes)) = read_mjpeg_frame(&mut reader, &mut buffered) {
                let frame = decode_preview_frame(&bytes);
                let failed = frame.is_err();
                let deadline = playback_frame_deadline(started_at, frame_index, frame_rate);
                std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
                if sender.send(frame).is_err() || failed {
                    break;
                }
                frame_index += 1;
            }
        });
    if let Err(error) = reader_thread {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("Could not start video frame reader: {error}"));
    }
    let first_frame = match frames.recv() {
        Ok(Ok(frame)) => frame,
        Ok(Err(error)) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Could not read first FFmpeg preview frame: {error}"
            ));
        }
    };
    Ok((
        PlaybackStream {
            child,
            frames,
            path: path.to_path_buf(),
            id,
            time: time + 1.0 / frame_rate,
            frame_rate,
        },
        first_frame,
    ))
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
    const ANALYSIS_MAX_DIMENSION: u32 = 320;
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
    let expected_frames = expected_motion_analysis_samples(duration);

    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-ss"])
        .arg(format!("{:.6}", project.trim_in))
        .arg("-i")
        .arg(&project.video_path)
        .arg("-t")
        .arg(format!("{duration:.6}"))
        .arg("-vf")
        .arg(format!(
            "fps={MOTION_ANALYSIS_FPS},scale={analysis_width}:{analysis_height}:flags=area,format=gray"
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
            let (delta_x, delta_y) =
                estimate_frame_translation(previous, &frame, analysis_width, analysis_height);
            cumulative_x = (cumulative_x + delta_x as f32).clamp(
                -(analysis_width as f32 * 0.15),
                analysis_width as f32 * 0.15,
            );
            cumulative_y = (cumulative_y + delta_y as f32).clamp(
                -(analysis_height as f32 * 0.15),
                analysis_height as f32 * 0.15,
            );
        }
        let sample_index = samples.len();
        samples.push(CameraMotionSample {
            time: project.trim_in + sample_index as f32 / MOTION_ANALYSIS_FPS as f32,
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

fn read_raw_frame(reader: &mut impl Read, frame_size: usize) -> std::io::Result<Option<Vec<u8>>> {
    let mut frame = vec![0_u8; frame_size];
    let first_read = reader.read(&mut frame[..1])?;
    if first_read == 0 {
        return Ok(None);
    }
    reader.read_exact(&mut frame[1..])?;
    Ok(Some(frame))
}

fn estimate_frame_translation(
    previous: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
) -> (i32, i32) {
    const SEARCH_RADIUS: i32 = 12;
    const SAMPLE_STEP: u32 = 4;
    const MAX_PIXEL_DIFFERENCE: u8 = 48;
    if width <= (SEARCH_RADIUS * 2) as u32 || height <= (SEARCH_RADIUS * 2) as u32 {
        return (0, 0);
    }

    let mut best = (u64::MAX, 0, 0);
    let mut second_best = u64::MAX;
    for delta_y in -SEARCH_RADIUS..=SEARCH_RADIUS {
        for delta_x in -SEARCH_RADIUS..=SEARCH_RADIUS {
            let mut error = 0_u64;
            let mut count = 0_u64;
            let y_start = SEARCH_RADIUS as u32;
            let y_end = height - SEARCH_RADIUS as u32;
            let x_start = SEARCH_RADIUS as u32;
            let x_end = width - SEARCH_RADIUS as u32;
            for y in (y_start..y_end).step_by(SAMPLE_STEP as usize) {
                for x in (x_start..x_end).step_by(SAMPLE_STEP as usize) {
                    let previous_index = (y * width + x) as usize;
                    let current_x = (x as i32 + delta_x) as u32;
                    let current_y = (y as i32 + delta_y) as u32;
                    let current_index = (current_y * width + current_x) as usize;
                    error += previous[previous_index]
                        .abs_diff(current[current_index])
                        .min(MAX_PIXEL_DIFFERENCE) as u64;
                    count += 1;
                }
            }
            let score = error / count.max(1);
            if score < best.0 {
                second_best = best.0;
                best = (score, delta_x, delta_y);
            } else if score < second_best {
                second_best = score;
            }
        }
    }

    if best.1.abs() == SEARCH_RADIUS
        || best.2.abs() == SEARCH_RADIUS
        || second_best.saturating_sub(best.0) < 1
    {
        (0, 0)
    } else {
        (best.1, best.2)
    }
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
    let mut points = trail_points(&project.keyframes, time, project.path_smoothing);
    let (offset_x, offset_y) = camera_motion_offset(project, time);
    for (x, y) in &mut points {
        *x += offset_x;
        *y += offset_y;
    }
    points
}

fn stop_preview_process(playback: &mut Option<PlaybackStream>) {
    if let Some(mut stream) = playback.take() {
        let _ = stream.child.kill();
        let _ = stream.child.wait();
    }
}

fn preview_stream_is_ready(
    stream_path: &Path,
    next_frame_time: f32,
    stream_frame_rate: f32,
    requested_path: &Path,
    requested_time: f32,
    frame_rate: f32,
) -> bool {
    if !frame_rate.is_finite()
        || frame_rate <= 0.0
        || stream_path != requested_path
        || (stream_frame_rate - frame_rate).abs() > 0.001
    {
        return false;
    }
    let first_expected_time = requested_time + 1.0 / frame_rate;
    let half_frame = 0.5 / frame_rate;
    let buffered_frames = 4.0;
    next_frame_time >= first_expected_time - half_frame
        && next_frame_time <= first_expected_time + buffered_frames / frame_rate
}

fn read_mjpeg_frame(
    reader: &mut impl Read,
    buffered: &mut Vec<u8>,
) -> std::io::Result<Option<Vec<u8>>> {
    const CHUNK_SIZE: usize = 8192;
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

fn probe_video_dimension(path: &Path, dimension: &str) -> Result<u32, String> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg(format!("stream={dimension}"))
        .args(["-of", "default=noprint_wrappers=1:nokey=1"])
        .arg(path)
        .output()
        .map_err(|error| format!("Could not start ffprobe: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Could not read video {dimension}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .map_err(|error| format!("ffprobe returned an invalid video {dimension}: {error}"))
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
    let mut last_report = Instant::now();
    for frame_index in 0..frames {
        let time = project.trim_in + frame_index as f32 / EXPORT_FPS as f32;
        let points = project_trail_points(project, time);
        let mut image = RgbaImage::new(width, height);
        let line_color = Rgba([rgb[0], rgb[1], rgb[2], 255]);
        let scaled: Vec<(f32, f32)> = points
            .iter()
            .map(|(x, y)| (x * width as f32, y * height as f32))
            .collect();
        draw_feathered_path(&mut image, &scaled, line_width, glow, rgb);
        if let Some((x, y)) = scaled.last().copied() {
            draw_filled_circle_mut(
                &mut image,
                (x.round() as i32, y.round() as i32),
                line_width.ceil() as i32,
                line_color,
            );
        }
        let frame_path = directory.join(format!("overlay_{frame_index:06}.png"));
        image
            .save_with_format(frame_path, ImageFormat::Png)
            .map_err(|error| format!("Could not render tracer frame: {error}"))?;
        if last_report.elapsed() >= Duration::from_millis(100) || frame_index + 1 == frames {
            let progress = (frame_index + 1) as f32 / frames as f32;
            report_progress(0.02 + progress * 0.48, "Rendering tracer");
            last_report = Instant::now();
        }
    }
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
    Some(RgbaImage::from_fn(width, height, |x, y| {
        let alpha = (blurred_mask.get_pixel(x, y).0[0] as f32 * glow_strength).round() as u8;
        Rgba([rgb[0], rgb[1], rgb[2], alpha])
    }))
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
    rgb: [u8; 3],
) {
    if let Some(glow_layer) =
        render_glow_layer(image.width(), image.height(), points, core_width, glow, rgb)
    {
        image::imageops::overlay(image, &glow_layer, 0, 0);
    }
    for segment in points.windows(2) {
        draw_thick_line(
            image,
            segment[0],
            segment[1],
            core_width.ceil() as u32,
            Rgba([rgb[0], rgb[1], rgb[2], 255]),
        );
    }
}

fn trail_points(keys: &[Keyframe], time: f32, smoothing: f32) -> Vec<(f32, f32)> {
    const CURVE_SAMPLES: usize = 16;
    let Some(first) = keys.first() else {
        return Vec::new();
    };
    if time < first.time {
        return Vec::new();
    }
    let smoothed_keys = smooth_keyframes(keys, smoothing);
    let keys = smoothed_keys.as_slice();
    let first = &keys[0];
    let mut points = vec![(first.x, first.y)];
    for segment_index in 0..keys.len().saturating_sub(1) {
        let start = &keys[segment_index];
        let end = &keys[segment_index + 1];
        let span = (end.time - start.time).max(f32::EPSILON);
        let fraction = ((time - start.time) / span).clamp(0.0, 1.0);
        if fraction <= 0.0 {
            break;
        }
        let p0 = &keys[segment_index.saturating_sub(1)];
        let p3 = &keys[(segment_index + 2).min(keys.len() - 1)];
        for sample in 1..=CURVE_SAMPLES {
            let sample_fraction = sample as f32 / CURVE_SAMPLES as f32;
            if sample_fraction >= fraction {
                break;
            }
            points.push(catmull_rom_point(p0, start, end, p3, sample_fraction));
        }
        points.push(catmull_rom_point(p0, start, end, p3, fraction));
        if fraction < 1.0 {
            break;
        }
    }
    points
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

fn encode_export(
    project: &Project,
    output: &Path,
    frame_dir: &Path,
    duration: f32,
    report_progress: &mut impl FnMut(f32, &str),
) -> Result<(), String> {
    let pattern = frame_dir.join("overlay_%06d.png");
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-ss"])
        .arg(format!("{:.3}", project.trim_in))
        .arg("-i")
        .arg(&project.video_path)
        .arg("-t")
        .arg(format!("{duration:.3}"))
        .arg("-framerate")
        .arg(EXPORT_FPS.to_string())
        .arg("-i")
        .arg(pattern)
        .args([
            "-filter_complex",
            "[0:v][1:v]overlay=shortest=1:format=auto[outv]",
            "-map",
            "[outv]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
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
        .map_err(|error| {
            format!("Could not start ffmpeg. Install FFmpeg and add it to PATH: {error}")
        })?;
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
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "OpenTracer PC",
        options,
        Box::new(|creation_context| Ok(Box::new(TracerApp::new(&creation_context.egui_ctx)))),
    )
}

fn apply_app_theme(ctx: &egui::Context) {
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
        camera_motion_offset, draw_feathered_path, estimate_frame_translation,
        expected_motion_analysis_samples, frame_time, internal_to_style_scale,
        playback_frame_deadline, preview_stream_is_ready, project_trail_points, read_mjpeg_frame,
        scaled_tracer_dimensions, smooth_keyframes, snap_trim_time, style_scale_to_internal,
        time_to_frame, timeline_range, trail_points, CameraMotionSample, Keyframe, Project,
        TracerStyle, TRACER_GLOW_MAX, TRACER_WIDTH_MAX,
    };
    use image::{Rgba, RgbaImage};
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

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
    fn tracer_is_hidden_before_first_keyframe() {
        let keys = [Keyframe {
            time: 1.0,
            x: 0.2,
            y: 0.4,
        }];
        assert!(trail_points(&keys, 0.5, 0.0).is_empty());
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
        let points = trail_points(&keys, 1.0, 0.0);
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
        let points = trail_points(&keys, 1.5, 0.0);
        assert!(points.iter().any(|(_, y)| *y > 0.5));
        let (x, y) = points.last().copied().expect("interpolated point");
        assert!((x - 0.5).abs() < 0.0001);
        assert!((y - 0.5625).abs() < 0.0001);
    }

    #[test]
    fn camera_motion_estimator_detects_small_scene_translations() {
        let (width, height) = (128_u32, 96_u32);
        let mut previous = vec![0_u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                previous[(y * width + x) as usize] = ((x * 73 + y * 29 + x * y * 17) % 251) as u8;
            }
        }
        let mut current = vec![0_u8; previous.len()];
        for y in 0..height - 2 {
            for x in 0..width - 3 {
                current[(y * width + x + 3) as usize] = previous[((y + 2) * width + x) as usize];
            }
        }
        assert_eq!(
            estimate_frame_translation(&previous, &current, width, height),
            (3, -2)
        );
    }

    #[test]
    fn ten_fps_motion_analysis_reports_samples_for_the_selected_duration() {
        assert_eq!(expected_motion_analysis_samples(23.56), 236);
        assert_eq!(expected_motion_analysis_samples(20.0), 200);
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
            path_smoothing: 0.5,
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
    fn paused_preview_stream_is_reused_only_for_nearby_frames() {
        let path = std::path::Path::new("/video.mov");
        let frame_rate = 30.0;
        let requested_time = 2.0;
        let next_frame_time = requested_time + 1.0 / frame_rate;
        assert!(preview_stream_is_ready(
            path,
            next_frame_time,
            frame_rate,
            path,
            requested_time,
            frame_rate,
        ));
        assert!(preview_stream_is_ready(
            path,
            next_frame_time + 3.0 / frame_rate,
            frame_rate,
            path,
            requested_time,
            frame_rate,
        ));
        assert!(!preview_stream_is_ready(
            path,
            next_frame_time + 5.0 / frame_rate,
            frame_rate,
            path,
            requested_time,
            frame_rate,
        ));
        assert!(!preview_stream_is_ready(
            path,
            next_frame_time,
            frame_rate,
            std::path::Path::new("/other.mov"),
            requested_time,
            frame_rate,
        ));
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
        );
        assert_eq!(image.get_pixel(30, 20), &Rgba([40, 220, 80, 255]));
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
}
