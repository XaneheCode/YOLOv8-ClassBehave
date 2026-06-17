const state = {
  config: null,
  view: "sender",
  mode: "person_vlm",
  running: false,
  yoloProcessing: false,
  visionProcessing: false,
  loopTimer: null,
  stream: null,
  objectUrl: null,
  lastVisionAt: 0,
  senderFrameCount: 0,
  latestFrame: null,
  latestYoloDetections: [],
};

const els = {
  viewTitle: document.querySelector("#viewTitle"),
  connectionPill: document.querySelector("#connectionPill"),
  connectionText: document.querySelector("#connectionText"),
  senderStatus: document.querySelector("#senderStatus"),
  sourceLabel: document.querySelector("#sourceLabel"),
  senderResolution: document.querySelector("#senderResolution"),
  senderFrames: document.querySelector("#senderFrames"),
  sourcePreviewImage: document.querySelector("#sourcePreviewImage"),
  sourceStage: document.querySelector(".source-stage"),
  sourceVideo: document.querySelector("#sourceVideo"),
  captureCanvas: document.querySelector("#captureCanvas"),
  sendInterval: document.querySelector("#sendInterval"),
  captureWidth: document.querySelector("#captureWidth"),
  jpegQuality: document.querySelector("#jpegQuality"),
  cameraSelect: document.querySelector("#cameraSelect"),
  refreshCameras: document.querySelector("#refreshCameras"),
  startCamera: document.querySelector("#startCamera"),
  pickImage: document.querySelector("#pickImage"),
  pickVideo: document.querySelector("#pickVideo"),
  stopSource: document.querySelector("#stopSource"),
  imageInput: document.querySelector("#imageInput"),
  videoInput: document.querySelector("#videoInput"),
  modelPath: document.querySelector("#modelPath"),
  alarmSeconds: document.querySelector("#alarmSeconds"),
  yoloEndpoint: document.querySelector("#yoloEndpoint"),
  yoloStatus: document.querySelector("#yoloStatus"),
  overlayImage: document.querySelector("#overlayImage"),
  videoStage: document.querySelector(".video-stage"),
  resolutionMetric: document.querySelector("#resolutionMetric"),
  latencyMetric: document.querySelector("#latencyMetric"),
  fpsMetric: document.querySelector("#fpsMetric"),
  alarmStrip: document.querySelector("#alarmStrip"),
  alarmText: document.querySelector("#alarmText"),
  alarmDetail: document.querySelector("#alarmDetail"),
  targetCount: document.querySelector("#targetCount"),
  abnormalCount: document.querySelector("#abnormalCount"),
  frameCount: document.querySelector("#frameCount"),
  countsList: document.querySelector("#countsList"),
  useVision: document.querySelector("#useVision"),
  uploadInterval: document.querySelector("#uploadInterval"),
  providerText: document.querySelector("#providerText"),
  providerChip: document.querySelector("#providerChip"),
  visionStatus: document.querySelector("#visionStatus"),
  visionMessage: document.querySelector("#visionMessage"),
  gridImage: document.querySelector("#gridImage"),
  cropPreview: document.querySelector(".crop-preview"),
  visionSummary: document.querySelector("#visionSummary"),
  peopleList: document.querySelector("#peopleList"),
  visionOverlayImage: document.querySelector("#visionOverlayImage"),
  visionStage: document.querySelector(".vision-stage"),
  visionResolution: document.querySelector("#visionResolution"),
  visionLatency: document.querySelector("#visionLatency"),
  visionFps: document.querySelector("#visionFps"),
  logList: document.querySelector("#logList"),
  clearLog: document.querySelector("#clearLog"),
  openAlarmDir: document.querySelector("#openAlarmDir"),
};

const viewTitles = {
  sender: "浏览器发送端",
  console: "实时分析控制台",
  vlm: "大模型分析模块",
  logs: "日志与设置",
};

const modeDefaults = {
  person_vlm: "yolov8s.pt",
  person_only: "yolov8s.pt",
  behaviour_yolo: "models/merged_classroom_6cls_v2_img960_e50_2026-06-13_best.pt",
};

const countColors = {
  "举手": "var(--blue)",
  "学习": "var(--green)",
  "使用手机": "var(--red)",
  "低头": "var(--orange)",
  "睡觉": "var(--purple)",
  person: "var(--muted)",
};

function setConnection(status, text) {
  els.connectionPill.classList.remove("ready", "error");
  if (status) els.connectionPill.classList.add(status);
  els.connectionText.textContent = text;
}

function log(message, kind = "info") {
  const item = document.createElement("div");
  item.className = `log-item ${kind}`;
  const now = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  item.innerHTML = `<time>${now}</time><strong>${message}</strong>`;
  els.logList.prepend(item);
  while (els.logList.children.length > 100) {
    els.logList.lastElementChild.remove();
  }
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.viewPanel === view);
  });
  els.viewTitle.textContent = viewTitles[view] || "课堂行为监测";
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (!config.ok) throw new Error("配置读取失败");
    state.config = config;
    state.mode = config.defaultMode;
    els.modelPath.value = config.personModel;
    els.yoloEndpoint.value = config.api?.yoloFrame || "/api/yolo-frame";
    els.uploadInterval.value = config.vision.intervalSeconds || 10;
    els.providerText.value = `${config.vision.provider} / ${config.vision.model}`;
    els.providerChip.textContent = `${config.vision.provider} · ${config.vision.model}`;
    renderCounts({});
    renderPeople([]);
    setConnection("ready", "Web API 已就绪");
    if (config.vision.configured) {
      log(`大模型接口已配置：${config.vision.provider} / ${config.vision.model}`);
    } else {
      log("未配置大模型 API Key，大模型模块会显示跳过或错误状态。", "warn");
    }
  } catch (error) {
    setConnection("error", "Web API 不可用");
    log(`读取配置失败：${error.message}`, "error");
  }
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  els.modelPath.value = modeDefaults[mode];
  log(`YOLO 模式切换为：${modeLabel(mode)}`);
}

function modeLabel(mode) {
  return {
    person_vlm: "人体检测",
    person_only: "仅人体检测",
    behaviour_yolo: "六类 YOLO",
  }[mode] || mode;
}

async function refreshCameraDevices({ silent = false } = {}) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    if (!silent) log("当前浏览器不支持摄像头设备枚举。", "warn");
    return;
  }

  try {
    const previousCameraId = els.cameraSelect.value;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === "videoinput");

    els.cameraSelect.innerHTML = '<option value="">默认摄像头</option>';
    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `摄像头 ${index + 1}`;
      els.cameraSelect.appendChild(option);
    });

    if (previousCameraId && cameras.some((camera) => camera.deviceId === previousCameraId)) {
      els.cameraSelect.value = previousCameraId;
    }

    if (!silent) {
      log(`已刷新摄像头列表：${cameras.length || 1} 个可用项`);
    }
  } catch (error) {
    if (!silent) log(`刷新摄像头失败：${error.message}`, "error");
  }
}

function stopSource() {
  state.running = false;
  if (state.loopTimer) {
    clearTimeout(state.loopTimer);
    state.loopTimer = null;
  }
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
  els.sourceVideo.pause();
  els.sourceVideo.removeAttribute("src");
  els.sourceVideo.srcObject = null;
  els.stopSource.disabled = true;
  els.senderStatus.textContent = "已停止";
  els.sourceLabel.textContent = "已停止输入";
  log("发送端已停止");
}

async function startCamera() {
  stopSource();
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前浏览器不支持摄像头采集");
    }
    const selectedCameraId = els.cameraSelect.value;
    const idealWidth = Math.max(320, Number(els.captureWidth.value || 1280));
    const videoConstraints = {
      width: { ideal: idealWidth },
      height: { ideal: Math.round(idealWidth * 9 / 16) },
    };
    if (selectedCameraId) {
      videoConstraints.deviceId = { exact: selectedCameraId };
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });
    state.stream = stream;
    els.sourceVideo.srcObject = stream;
    await els.sourceVideo.play();
    await refreshCameraDevices({ silent: true });
    const cameraLabel = stream.getVideoTracks()[0]?.label || "浏览器摄像头";
    state.running = true;
    els.stopSource.disabled = false;
    els.senderStatus.textContent = "摄像头发送中";
    els.sourceLabel.textContent = cameraLabel;
    log(`摄像头已启动：${cameraLabel}，开始发送到 YOLO 控制台。`);
    scheduleLoop(true);
  } catch (error) {
    els.senderStatus.textContent = "摄像头失败";
    log(`启动摄像头失败：${error.message}`, "error");
  }
}

function pickImage() {
  els.imageInput.click();
}

function pickVideo() {
  els.videoInput.click();
}

function handleImageFile(file) {
  stopSource();
  const reader = new FileReader();
  reader.onload = async () => {
    const dataUrl = String(reader.result);
    state.latestFrame = dataUrl;
    updateSourcePreview(dataUrl, `图片测试：${file.name}`);
    els.senderStatus.textContent = "图片已发送";
    log(`发送图片到 YOLO 控制台：${file.name}`);
    await sendFrame(dataUrl);
  };
  reader.readAsDataURL(file);
}

async function handleVideoFile(file) {
  stopSource();
  state.objectUrl = URL.createObjectURL(file);
  els.sourceVideo.src = state.objectUrl;
  els.sourceVideo.loop = true;
  els.sourceVideo.muted = true;
  await els.sourceVideo.play();
  state.running = true;
  els.stopSource.disabled = false;
  els.senderStatus.textContent = "视频发送中";
  els.sourceLabel.textContent = `视频测试：${file.name}`;
  log(`视频已载入，开始抽帧发送：${file.name}`);
  scheduleLoop(true);
}

function scheduleLoop(immediate = false) {
  if (!state.running) return;
  const delay = immediate ? 0 : Math.max(300, Number(els.sendInterval.value || 1) * 1000);
  state.loopTimer = setTimeout(async () => {
    if (state.running && !state.yoloProcessing) {
      const frame = captureVideoFrame();
      if (frame) {
        await sendFrame(frame);
      }
    }
    scheduleLoop(false);
  }, delay);
}

function captureVideoFrame() {
  const video = els.sourceVideo;
  if (!video.videoWidth || !video.videoHeight) return null;
  const maxWidth = Number(els.captureWidth.value || 1280);
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const width = Math.round(video.videoWidth * scale);
  const height = Math.round(video.videoHeight * scale);
  els.captureCanvas.width = width;
  els.captureCanvas.height = height;
  const context = els.captureCanvas.getContext("2d");
  context.drawImage(video, 0, 0, width, height);
  return els.captureCanvas.toDataURL("image/jpeg", Number(els.jpegQuality.value || 86) / 100);
}

async function sendFrame(dataUrl) {
  state.latestFrame = dataUrl;
  updateSourcePreview(dataUrl);
  state.senderFrameCount += 1;
  els.senderFrames.textContent = `${state.senderFrameCount} 帧`;
  await analyzeYolo(dataUrl);
}

function updateSourcePreview(dataUrl, label) {
  els.sourcePreviewImage.src = dataUrl;
  els.sourceStage.classList.add("has-image");
  if (label) els.sourceLabel.textContent = label;
  const image = new Image();
  image.onload = () => {
    els.senderResolution.textContent = `${image.naturalWidth}×${image.naturalHeight}`;
  };
  image.src = dataUrl;
}

async function analyzeYolo(dataUrl) {
  if (state.yoloProcessing) return;
  state.yoloProcessing = true;
  setConnection("ready", "YOLO 分析中");
  els.yoloStatus.textContent = "YOLO 分析中";
  try {
    const response = await fetch("/api/yolo-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: dataUrl,
        mode: state.mode,
        modelPath: els.modelPath.value.trim() || modeDefaults[state.mode],
        alarmSeconds: Number(els.alarmSeconds.value || 3),
      }),
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "YOLO 分析失败");
    state.latestYoloDetections = result.yoloDetections || [];
    renderYolo(result);
    setConnection("ready", "YOLO 已更新");
    maybeAnalyzeVision(dataUrl, state.latestYoloDetections);
  } catch (error) {
    setConnection("error", "YOLO 分析失败");
    els.yoloStatus.textContent = "YOLO 失败";
    log(`YOLO 分析失败：${error.message}`, "error");
  } finally {
    state.yoloProcessing = false;
  }
}

function maybeAnalyzeVision(dataUrl, detections) {
  if (!els.useVision.checked) {
    els.visionStatus.textContent = "大模型已关闭";
    return;
  }
  if (state.mode !== "person_vlm" && state.mode !== "person_only") {
    els.visionStatus.textContent = "六类 YOLO 模式不上传大模型";
    return;
  }
  if (!detections.length) {
    els.visionStatus.textContent = "没有 YOLO 人体框";
    return;
  }
  if (state.visionProcessing) {
    els.visionStatus.textContent = "上一帧大模型分析中，本帧不上传";
    return;
  }
  const intervalMs = Math.max(1000, Number(els.uploadInterval.value || 10) * 1000);
  const now = Date.now();
  if (now - state.lastVisionAt < intervalMs) return;
  state.lastVisionAt = now;
  analyzeVision(dataUrl, detections);
}

async function analyzeVision(dataUrl, detections) {
  state.visionProcessing = true;
  els.visionStatus.textContent = "大模型分析中";
  els.visionMessage.textContent = "正在上传编号人体拼图";
  try {
    const response = await fetch("/api/vlm-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: dataUrl,
        detections,
        alarmSeconds: Number(els.alarmSeconds.value || 3),
        useVision: els.useVision.checked,
      }),
    });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "大模型分析失败");
    renderVision(result);
    log(`大模型模块更新：${result.vision?.message || "完成"}`);
  } catch (error) {
    els.visionStatus.textContent = "大模型分析失败";
    els.visionMessage.textContent = error.message;
    log(`大模型分析失败：${error.message}`, "error");
  } finally {
    state.visionProcessing = false;
  }
}

function renderYolo(result) {
  els.overlayImage.src = result.overlayImage;
  els.videoStage.classList.add("has-image");
  els.resolutionMetric.textContent = `${result.frame.width}×${result.frame.height}`;
  els.latencyMetric.textContent = `${result.frame.latencyMs} ms`;
  els.fpsMetric.textContent = `${result.frame.fps} FPS`;
  els.targetCount.textContent = result.targetCount;
  els.abnormalCount.textContent = result.alarm.abnormalCount;
  els.frameCount.textContent = result.frame.frameCount;
  els.yoloStatus.textContent = `${modeLabel(result.mode)} · ${result.targetCount} 个目标`;
  updateAlarm(result.alarm);
  renderCounts(result.counts || {});
  const alarmText = result.alarm.isAlarm ? "报警" : result.alarm.suspicious ? "疑似异常" : "正常";
  log(`YOLO ${alarmText} · ${result.targetCount} 个目标 · ${modeLabel(result.mode)}`);
  if (result.alarmRecord) {
    log(`YOLO 报警已保存：${result.alarmRecord.imagePath}`);
  }
}

function renderVision(result) {
  const vision = result.vision || {};
  els.visionStatus.textContent = vision.message || "大模型完成";
  els.visionMessage.textContent = vision.message || "无状态";
  els.visionSummary.textContent = vision.summary || "暂无结果";
  els.visionResolution.textContent = `${result.frame.width}×${result.frame.height}`;
  els.visionLatency.textContent = `${result.frame.latencyMs} ms`;
  els.visionFps.textContent = `${result.frame.fps} FPS`;

  if (result.overlayImage) {
    els.visionOverlayImage.src = result.overlayImage;
    els.visionStage.classList.add("has-image");
  }
  if (vision.gridImage) {
    els.gridImage.src = vision.gridImage;
    els.cropPreview.classList.add("has-image");
  }
  if (result.alarmRecord) {
    log(`大模型报警已保存：${result.alarmRecord.imagePath}`);
  }
  renderPeople(vision.people || []);
}

function updateAlarm(alarm) {
  els.alarmStrip.classList.remove("alarm", "suspicious");
  if (alarm.isAlarm) {
    els.alarmStrip.classList.add("alarm");
    els.alarmText.textContent = "异常报警";
  } else if (alarm.suspicious) {
    els.alarmStrip.classList.add("suspicious");
    els.alarmText.textContent = "疑似异常";
  } else {
    els.alarmText.textContent = "正常";
  }
  const labels = (alarm.abnormalLabels || []).join("、") || "无异常目标";
  els.alarmDetail.textContent = `${labels} · 持续 ${alarm.durationSeconds || 0}s`;
}

function renderCounts(counts) {
  const order = ["person", "举手", "学习", "使用手机", "低头", "睡觉"];
  els.countsList.innerHTML = "";
  order.forEach((label) => {
    if (!(label in counts) && label !== "person") return;
    const value = counts[label] || 0;
    const row = document.createElement("div");
    row.className = "count-row";
    row.innerHTML = `
      <span class="dot" style="background:${countColors[label] || "var(--muted)"}"></span>
      <strong>${label}</strong>
      <span>${value}</span>
    `;
    els.countsList.appendChild(row);
  });
}

function renderPeople(people) {
  els.peopleList.innerHTML = "";
  if (!people.length) {
    const empty = document.createElement("div");
    empty.className = "person-row";
    empty.innerHTML = '<div class="person-id">--</div><div><strong>暂无大模型分类结果</strong><span>等待大模型模块返回</span></div><div class="confidence">--</div>';
    els.peopleList.appendChild(empty);
    return;
  }
  people.forEach((person, index) => {
    const row = document.createElement("div");
    row.className = "person-row";
    row.innerHTML = `
      <div class="person-id">${index + 1}</div>
      <div>
        <strong>${person.displayLabel || person.label}</strong>
        <span>${person.status || "unknown"} · [${(person.bbox || []).join(", ")}]</span>
      </div>
      <div class="confidence">${person.confidence || "unknown"}</div>
    `;
    els.peopleList.appendChild(row);
  });
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

els.startCamera.addEventListener("click", startCamera);
els.refreshCameras.addEventListener("click", () => refreshCameraDevices());
els.stopSource.addEventListener("click", stopSource);
els.pickImage.addEventListener("click", () => els.imageInput.click());
els.pickVideo.addEventListener("click", () => els.videoInput.click());
els.imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) handleImageFile(file);
  event.target.value = "";
});
els.videoInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) handleVideoFile(file);
  event.target.value = "";
});
els.clearLog.addEventListener("click", () => {
  els.logList.innerHTML = "";
});
els.openAlarmDir.addEventListener("click", () => {
  log("报警文件默认保存在 output/alarms。浏览器不能直接打开本地目录。", "warn");
});

const initialView = new URLSearchParams(window.location.search).get("view");
if (initialView && viewTitles[initialView]) {
  switchView(initialView);
}

loadConfig();
refreshCameraDevices({ silent: true });
