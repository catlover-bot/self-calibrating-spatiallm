const SPLIT_DEFS = [
  {
    id: "small",
    label: "Small Benchmark",
    basePath: "../outputs/eval_pack/latest",
  },
  {
    id: "public",
    label: "Public Medium Benchmark",
    basePath: "../outputs/eval_pack/public_medium_latest",
  },
];

const SETTING_ORDER = [
  "no_calibration",
  "calibration_v0",
  "calibration_v1",
  "calibration_v1_plus_repair",
  "mock_generator",
  "external_generator",
];

const SETTING_LABELS = {
  no_calibration: "No Calibration",
  calibration_v0: "Calibration v0",
  calibration_v1: "Calibration v1",
  calibration_v1_plus_repair: "Calibration v1 + Repair",
  mock_generator: "Mock Generator",
  external_generator: "External Generator",
};

const DELTA_ALIAS = {
  calibration_v0: "v0",
  calibration_v1: "v1",
  calibration_v1_plus_repair: "v1_plus_repair",
  mock_generator: "mock_generator",
  external_generator: "external_generator",
};

const CANVAS_MAX_OBJECTS = 42;
const QA_LIST_LIMIT = 8;
const GROUNDING_LIST_LIMIT = 8;

const state = {
  splitId: "small",
  sceneId: null,
  leftSetting: "calibration_v0",
  rightSetting: "calibration_v1",
  splitBundles: new Map(),
  presets: [],
  compiledSteps: [],
  autoplay: {
    running: false,
    timer: null,
    stepIndex: 0,
  },
  statusMessages: [],
};

const el = {
  splitSelect: document.getElementById("splitSelect"),
  sceneSelect: document.getElementById("sceneSelect"),
  leftSettingSelect: document.getElementById("leftSettingSelect"),
  rightSettingSelect: document.getElementById("rightSettingSelect"),
  presetSelect: document.getElementById("presetSelect"),
  stepDurationInput: document.getElementById("stepDurationInput"),
  loopCheckbox: document.getElementById("loopCheckbox"),
  autoplayToggleBtn: document.getElementById("autoplayToggleBtn"),
  nextStepBtn: document.getElementById("nextStepBtn"),
  stepCounter: document.getElementById("stepCounter"),
  stepDescription: document.getElementById("stepDescription"),
  statusBanner: document.getElementById("statusBanner"),
  diffObjectLabels: document.getElementById("diffObjectLabels"),
  diffRelationPredicates: document.getElementById("diffRelationPredicates"),
  diffCounts: document.getElementById("diffCounts"),
  narrationList: document.getElementById("narrationList"),
};

const panel = {
  left: {
    title: document.getElementById("leftTitle"),
    canvas: document.getElementById("leftCanvas"),
    evidenceChip: document.getElementById("leftEvidenceChip"),
    sourceChip: document.getElementById("leftSourceChip"),
    statusChip: document.getElementById("leftStatusChip"),
    sceneSummary: document.getElementById("leftSceneSummary"),
    relationText: document.getElementById("leftRelationText"),
    objectsList: document.getElementById("leftObjectsList"),
    relationsList: document.getElementById("leftRelationsList"),
    qaList: document.getElementById("leftQaList"),
    groundingList: document.getElementById("leftGroundingList"),
    provenance: document.getElementById("leftProvenance"),
  },
  right: {
    title: document.getElementById("rightTitle"),
    canvas: document.getElementById("rightCanvas"),
    evidenceChip: document.getElementById("rightEvidenceChip"),
    sourceChip: document.getElementById("rightSourceChip"),
    statusChip: document.getElementById("rightStatusChip"),
    sceneSummary: document.getElementById("rightSceneSummary"),
    relationText: document.getElementById("rightRelationText"),
    objectsList: document.getElementById("rightObjectsList"),
    relationsList: document.getElementById("rightRelationsList"),
    qaList: document.getElementById("rightQaList"),
    groundingList: document.getElementById("rightGroundingList"),
    provenance: document.getElementById("rightProvenance"),
  },
};

init().catch((error) => {
  console.error(error);
  setStatusBanner([
    "Demo initialization failed.",
    String(error?.message || error),
  ]);
});

async function init() {
  bindEvents();
  await loadPresets();
  await loadSplitBundles();
  initializeControls();
  applyQueryOverrides();
  updatePresetForSplit();
  refreshRender();
}

function bindEvents() {
  el.splitSelect.addEventListener("change", () => {
    stopAutoplay();
    state.splitId = el.splitSelect.value;
    syncSceneOptions();
    syncSettingOptions();
    updatePresetForSplit();
    refreshRender();
  });

  el.sceneSelect.addEventListener("change", () => {
    stopAutoplay();
    state.sceneId = el.sceneSelect.value;
    syncSettingOptions();
    refreshRender();
  });

  el.leftSettingSelect.addEventListener("change", () => {
    stopAutoplay();
    state.leftSetting = el.leftSettingSelect.value;
    refreshRender();
  });

  el.rightSettingSelect.addEventListener("change", () => {
    stopAutoplay();
    state.rightSetting = el.rightSettingSelect.value;
    refreshRender();
  });

  el.presetSelect.addEventListener("change", () => {
    stopAutoplay();
    compileSelectedPreset();
    refreshRender();
  });

  el.autoplayToggleBtn.addEventListener("click", () => {
    if (state.autoplay.running) {
      stopAutoplay();
    } else {
      startAutoplay();
    }
  });

  el.nextStepBtn.addEventListener("click", () => {
    if (!state.compiledSteps.length) {
      return;
    }
    if (!state.autoplay.running) {
      state.autoplay.stepIndex = Math.min(state.autoplay.stepIndex, state.compiledSteps.length - 1);
    }
    applyAutoplayStep(state.autoplay.stepIndex);
    state.autoplay.stepIndex += 1;
    if (state.autoplay.stepIndex >= state.compiledSteps.length) {
      if (el.loopCheckbox.checked) {
        state.autoplay.stepIndex = 0;
      } else {
        state.autoplay.stepIndex = state.compiledSteps.length - 1;
      }
    }
    updateAutoplayStatus();
  });

  window.addEventListener("keydown", (event) => {
    if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) {
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      if (state.autoplay.running) {
        stopAutoplay();
      } else {
        startAutoplay();
      }
    }
    if (event.code === "ArrowRight") {
      event.preventDefault();
      el.nextStepBtn.click();
    }
  });
}

async function loadPresets() {
  const presets = await fetchJsonMaybe("./presets.json");
  state.presets = Array.isArray(presets) ? presets : [];
}

async function loadSplitBundles() {
  const bundles = await Promise.all(SPLIT_DEFS.map((splitDef) => loadSingleSplitBundle(splitDef)));
  for (const bundle of bundles) {
    state.splitBundles.set(bundle.id, bundle);
  }
}

async function loadSingleSplitBundle(splitDef) {
  const bundle = {
    ...splitDef,
    available: false,
    report: null,
    scenes: [],
    sceneById: new Map(),
    sceneOrder: [],
    sceneDeltaById: new Map(),
    language: {
      sceneByKey: new Map(),
      qaByKey: new Map(),
      groundingByKey: new Map(),
      alignmentByScene: new Map(),
      summary: null,
    },
    warnings: [],
    errors: [],
  };

  const report = await fetchJsonMaybe(`${splitDef.basePath}/evaluation_report.json`);
  if (!report || typeof report !== "object") {
    bundle.errors.push(
      `Missing evaluation_report.json at ${splitDef.basePath}. This split is unavailable in the demo.`
    );
    return bundle;
  }
  bundle.report = report;
  bundle.available = true;

  const scenes = Array.isArray(report.scenes) ? report.scenes : [];
  for (const rawScene of scenes) {
    if (!rawScene || typeof rawScene !== "object") {
      continue;
    }
    const sceneId = safeString(rawScene.scene_id) || `scene_${bundle.scenes.length + 1}`;
    const settingsArray = Array.isArray(rawScene.settings) ? rawScene.settings : [];
    const settingsMap = new Map();
    for (const settingEntry of settingsArray) {
      if (!settingEntry || typeof settingEntry !== "object") {
        continue;
      }
      const settingName = safeString(settingEntry.setting_name);
      if (!settingName) {
        continue;
      }
      settingsMap.set(settingName, settingEntry);
    }

    const sceneObj = {
      sceneId,
      sourceType: safeString(rawScene.source_type) || "unknown",
      tags: Array.isArray(rawScene.tags) ? rawScene.tags : [],
      sampleConfigPath: safeString(rawScene.sample_config_path) || null,
      settingsMap,
      settingNames: orderedSettings(Array.from(settingsMap.keys())),
      deltaRow: null,
      predictionCache: new Map(),
    };

    bundle.scenes.push(sceneObj);
    bundle.sceneById.set(sceneId, sceneObj);
    bundle.sceneOrder.push(sceneId);
  }

  const sceneDeltaRows = await fetchJsonMaybe(`${splitDef.basePath}/scene_level_delta_report.json`);
  if (Array.isArray(sceneDeltaRows)) {
    for (const row of sceneDeltaRows) {
      if (!row || typeof row !== "object") {
        continue;
      }
      const sceneId = safeString(row.scene_id);
      if (!sceneId) {
        continue;
      }
      bundle.sceneDeltaById.set(sceneId, row);
      const sceneObj = bundle.sceneById.get(sceneId);
      if (sceneObj) {
        sceneObj.deltaRow = row;
      }
    }
  }

  await loadLanguageArtifacts(bundle);
  await hydratePredictionCaches(bundle);

  if (!bundle.scenes.length) {
    bundle.warnings.push(`No scenes found in ${splitDef.basePath}/evaluation_report.json.`);
  }
  return bundle;
}

async function loadLanguageArtifacts(bundle) {
  const base = `${bundle.basePath}/language`;

  const [sceneRows, qaRows, groundingRows, alignmentRows, summaryPayload] = await Promise.all([
    fetchJsonlMaybe(`${base}/language_scene_examples.jsonl`),
    fetchJsonlMaybe(`${base}/language_qa_examples.jsonl`),
    fetchJsonlMaybe(`${base}/language_grounding_examples.jsonl`),
    fetchJsonlMaybe(`${base}/language_alignment_examples.jsonl`),
    fetchJsonMaybe(`${base}/language_export_summary.json`),
  ]);

  if (!sceneRows.length) {
    bundle.warnings.push(`Language JSONL not found under ${base}; demo will use evaluation summaries.`);
  }

  bundle.language.summary = summaryPayload && typeof summaryPayload === "object" ? summaryPayload : null;

  for (const row of sceneRows) {
    if (!row || typeof row !== "object") {
      continue;
    }
    const sceneId = safeString(row.scene_id);
    const setting = safeString(row.setting);
    if (!sceneId || !setting) {
      continue;
    }
    bundle.language.sceneByKey.set(makeSceneSettingKey(sceneId, setting), row);
  }

  for (const row of qaRows) {
    if (!row || typeof row !== "object") {
      continue;
    }
    const sceneId = safeString(row.scene_id);
    const setting = safeString(row.setting);
    if (!sceneId || !setting) {
      continue;
    }
    const key = makeSceneSettingKey(sceneId, setting);
    const items = bundle.language.qaByKey.get(key) || [];
    items.push(row);
    bundle.language.qaByKey.set(key, items);
  }

  for (const row of groundingRows) {
    if (!row || typeof row !== "object") {
      continue;
    }
    const sceneId = safeString(row.scene_id);
    const setting = safeString(row.setting);
    if (!sceneId || !setting) {
      continue;
    }
    const key = makeSceneSettingKey(sceneId, setting);
    const items = bundle.language.groundingByKey.get(key) || [];
    items.push(row);
    bundle.language.groundingByKey.set(key, items);
  }

  for (const row of alignmentRows) {
    if (!row || typeof row !== "object") {
      continue;
    }
    const sceneId = safeString(row.scene_id);
    if (!sceneId) {
      continue;
    }
    const items = bundle.language.alignmentByScene.get(sceneId) || [];
    items.push(row);
    bundle.language.alignmentByScene.set(sceneId, items);
  }
}

async function hydratePredictionCaches(bundle) {
  for (const sceneObj of bundle.scenes) {
    for (const settingName of sceneObj.settingNames) {
      const settingEntry = sceneObj.settingsMap.get(settingName);
      const languageRow = bundle.language.sceneByKey.get(makeSceneSettingKey(sceneObj.sceneId, settingName));
      const resolved = await resolveStructuredPredictionPayload({
        bundle,
        sceneObj,
        settingName,
        settingEntry,
        languageRow,
      });
      sceneObj.predictionCache.set(settingName, resolved);
    }
  }
}

async function resolveStructuredPredictionPayload({ bundle, sceneObj, settingName, settingEntry, languageRow }) {
  const trace = [];
  if (languageRow && isPredictionPayload(languageRow.structured_prediction)) {
    trace.push("language_scene_examples.structured_prediction");
    return {
      payload: languageRow.structured_prediction,
      source: "language_scene_examples",
      trace,
    };
  }

  const metadata = settingEntry && typeof settingEntry === "object" && settingEntry.metadata && typeof settingEntry.metadata === "object"
    ? settingEntry.metadata
    : {};

  for (const key of [
    "structured_prediction_pre_repair",
    "scene_prediction",
    "structured_prediction",
    "prediction",
  ]) {
    if (isPredictionPayload(metadata[key])) {
      trace.push(`setting.metadata.${key}`);
      return {
        payload: metadata[key],
        source: `setting.metadata.${key}`,
        trace,
      };
    }
  }

  const artifactPaths = collectPredictionArtifactPaths(metadata);
  for (const relativePath of artifactPaths) {
    const resolvedUrl = resolveArtifactUrl(bundle.basePath, relativePath);
    if (!resolvedUrl) {
      trace.push(`artifact_path_unusable:${relativePath}`);
      continue;
    }
    const payload = await fetchJsonMaybe(resolvedUrl);
    if (!payload) {
      trace.push(`artifact_not_found:${relativePath}`);
      continue;
    }
    const predictionPayload = extractPredictionPayload(payload);
    if (!predictionPayload) {
      trace.push(`artifact_not_prediction_payload:${relativePath}`);
      continue;
    }
    trace.push(`artifact_path:${relativePath}`);
    return {
      payload: predictionPayload,
      source: `artifact:${relativePath}`,
      trace,
    };
  }

  if (sceneObj.deltaRow) {
    const alias = DELTA_ALIAS[settingName];
    if (alias) {
      const summary = sceneObj.deltaRow[`prediction_summary_${alias}`];
      if (summary && typeof summary === "object") {
        trace.push(`scene_level_delta_report.prediction_summary_${alias}`);
      }
    }
  }

  trace.push("summary_fallback");
  return {
    payload: null,
    source: "summary_fallback",
    trace,
  };
}

function collectPredictionArtifactPaths(metadata) {
  const paths = [];

  const directKeys = [
    "structured_prediction_pre_repair_path",
    "structured_prediction_path",
    "scene_prediction_path",
    "prediction_artifact_path",
    "prediction_path",
  ];

  for (const key of directKeys) {
    const value = metadata[key];
    if (typeof value === "string" && value.trim()) {
      paths.push(value.trim());
    }
  }

  if (metadata.prediction_artifact_paths && typeof metadata.prediction_artifact_paths === "object") {
    for (const value of Object.values(metadata.prediction_artifact_paths)) {
      if (typeof value === "string" && value.trim()) {
        paths.push(value.trim());
      }
    }
  }

  if (metadata.prediction_source_contract && typeof metadata.prediction_source_contract === "object") {
    const sourceContract = metadata.prediction_source_contract;
    if (sourceContract.sources && typeof sourceContract.sources === "object") {
      for (const sourceValue of Object.values(sourceContract.sources)) {
        if (sourceValue && typeof sourceValue === "object") {
          const pathValue = sourceValue.path;
          if (typeof pathValue === "string" && pathValue.trim()) {
            paths.push(pathValue.trim());
          }
        }
      }
    }
  }

  return Array.from(new Set(paths));
}

function resolveArtifactUrl(basePath, candidatePath) {
  if (!candidatePath || typeof candidatePath !== "string") {
    return null;
  }
  const trimmed = candidatePath.trim();
  if (!trimmed) {
    return null;
  }

  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }

  if (trimmed.startsWith("/")) {
    // Absolute filesystem paths are not directly fetchable from the static demo server.
    return null;
  }

  if (trimmed.startsWith("outputs/") || trimmed.startsWith("configs/") || trimmed.startsWith("data/")) {
    return `../${trimmed}`;
  }

  return `${basePath}/${trimmed}`;
}

function extractPredictionPayload(payload) {
  if (isPredictionPayload(payload)) {
    return payload;
  }
  if (payload && typeof payload === "object") {
    for (const key of [
      "scene_prediction",
      "structured_prediction",
      "prediction",
      "structured_prediction_pre_repair",
      "result",
    ]) {
      if (isPredictionPayload(payload[key])) {
        return payload[key];
      }
    }
  }
  return null;
}

function isPredictionPayload(payload) {
  return (
    !!payload &&
    typeof payload === "object" &&
    Array.isArray(payload.objects) &&
    Array.isArray(payload.relations)
  );
}

function initializeControls() {
  populateSplitOptions();
  syncSceneOptions();
  syncSettingOptions();
  populatePresetOptions();
  compileSelectedPreset();
}

function populateSplitOptions() {
  clearChildren(el.splitSelect);
  for (const splitDef of SPLIT_DEFS) {
    const bundle = state.splitBundles.get(splitDef.id);
    const option = document.createElement("option");
    option.value = splitDef.id;
    const unavailable = bundle && !bundle.available;
    option.textContent = unavailable ? `${splitDef.label} (unavailable)` : splitDef.label;
    el.splitSelect.appendChild(option);
  }

  const currentBundle = state.splitBundles.get(state.splitId);
  if (!currentBundle || !currentBundle.available) {
    const firstAvailable = SPLIT_DEFS.find((splitDef) => {
      const bundle = state.splitBundles.get(splitDef.id);
      return bundle && bundle.available;
    });
    state.splitId = firstAvailable ? firstAvailable.id : SPLIT_DEFS[0].id;
  }
  el.splitSelect.value = state.splitId;
}

function syncSceneOptions() {
  const bundle = getCurrentBundle();
  clearChildren(el.sceneSelect);

  if (!bundle || !bundle.available || !bundle.sceneOrder.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No scenes available";
    el.sceneSelect.appendChild(option);
    state.sceneId = null;
    return;
  }

  for (const sceneId of bundle.sceneOrder) {
    const sceneObj = bundle.sceneById.get(sceneId);
    const option = document.createElement("option");
    option.value = sceneId;
    const tagText = sceneObj && sceneObj.tags.length ? ` [${sceneObj.tags.slice(0, 2).join(", ")}]` : "";
    option.textContent = `${sceneId}${tagText}`;
    el.sceneSelect.appendChild(option);
  }

  if (!state.sceneId || !bundle.sceneById.has(state.sceneId)) {
    state.sceneId = bundle.sceneOrder[0];
  }
  el.sceneSelect.value = state.sceneId;
}

function syncSettingOptions() {
  const sceneObj = getCurrentScene();
  const settingNames = sceneObj ? sceneObj.settingNames : [];

  for (const select of [el.leftSettingSelect, el.rightSettingSelect]) {
    clearChildren(select);
    for (const settingName of settingNames) {
      const option = document.createElement("option");
      option.value = settingName;
      option.textContent = SETTING_LABELS[settingName] || settingName;
      select.appendChild(option);
    }
  }

  if (!settingNames.length) {
    state.leftSetting = "";
    state.rightSetting = "";
    return;
  }

  if (!settingNames.includes(state.leftSetting)) {
    state.leftSetting = settingNames.includes("calibration_v0")
      ? "calibration_v0"
      : settingNames[0];
  }
  if (!settingNames.includes(state.rightSetting)) {
    state.rightSetting = settingNames.includes("calibration_v1")
      ? "calibration_v1"
      : settingNames[Math.min(1, settingNames.length - 1)];
  }
  if (state.leftSetting === state.rightSetting && settingNames.length > 1) {
    const alternative = settingNames.find((name) => name !== state.leftSetting);
    if (alternative) {
      state.rightSetting = alternative;
    }
  }

  el.leftSettingSelect.value = state.leftSetting;
  el.rightSettingSelect.value = state.rightSetting;
}

function populatePresetOptions() {
  clearChildren(el.presetSelect);
  const presets = state.presets.length
    ? state.presets
    : [{ id: "manual", title: "Manual only", split: state.splitId, steps: [] }];

  for (const preset of presets) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = preset.title || preset.id;
    el.presetSelect.appendChild(option);
  }

  if (!el.presetSelect.value) {
    el.presetSelect.value = presets[0]?.id || "";
  }
}

function updatePresetForSplit() {
  const splitId = state.splitId;
  const currentPreset = getSelectedPreset();
  if (currentPreset && currentPreset.split === splitId) {
    compileSelectedPreset();
    return;
  }

  const match = state.presets.find((preset) => preset.split === splitId) || state.presets[0];
  if (match) {
    el.presetSelect.value = match.id;
  }
  compileSelectedPreset();
}

function compileSelectedPreset() {
  const preset = getSelectedPreset();
  const bundle = getCurrentBundle();
  if (!preset || !bundle || !bundle.available) {
    state.compiledSteps = [];
    state.autoplay.stepIndex = 0;
    updateAutoplayStatus();
    return;
  }

  const steps = compilePresetSteps(preset, bundle);
  state.compiledSteps = steps;
  state.autoplay.stepIndex = 0;

  if (preset.default_step_seconds && Number.isFinite(Number(preset.default_step_seconds))) {
    el.stepDurationInput.value = String(Number(preset.default_step_seconds));
  }
  updateAutoplayStatus();
}

function compilePresetSteps(preset, bundle) {
  const steps = [];
  const sceneExists = (sceneId) => bundle.sceneById.has(sceneId);

  if (Array.isArray(preset.steps) && preset.steps.length) {
    for (const step of preset.steps) {
      if (!step || typeof step !== "object") {
        continue;
      }
      const sceneId = safeString(step.scene_id);
      const leftSetting = safeString(step.left_setting);
      const rightSetting = safeString(step.right_setting);
      if (!sceneId || !leftSetting || !rightSetting) {
        continue;
      }
      if (!sceneExists(sceneId)) {
        continue;
      }
      const sceneObj = bundle.sceneById.get(sceneId);
      if (!sceneObj.settingNames.includes(leftSetting) || !sceneObj.settingNames.includes(rightSetting)) {
        continue;
      }
      steps.push({
        split: bundle.id,
        scene_id: sceneId,
        left_setting: leftSetting,
        right_setting: rightSetting,
        focus: safeString(step.focus) || preset.description || "",
        talking_points: Array.isArray(step.talking_points) ? step.talking_points : [],
      });
    }
  }

  if (!steps.length && preset.strategy && typeof preset.strategy === "object") {
    const strategyType = safeString(preset.strategy.type);
    if (strategyType === "first_n_scenes") {
      const count = Math.max(1, Number(preset.strategy.count) || 3);
      const comparisonsRaw = Array.isArray(preset.strategy.comparisons)
        ? preset.strategy.comparisons
        : [["calibration_v0", "calibration_v1"]];
      const comparisons = comparisonsRaw
        .filter((pair) => Array.isArray(pair) && pair.length >= 2)
        .map((pair) => [safeString(pair[0]), safeString(pair[1])])
        .filter((pair) => pair[0] && pair[1]);

      for (const sceneId of bundle.sceneOrder.slice(0, count)) {
        const sceneObj = bundle.sceneById.get(sceneId);
        if (!sceneObj) {
          continue;
        }
        for (const [leftSetting, rightSetting] of comparisons) {
          if (!sceneObj.settingNames.includes(leftSetting) || !sceneObj.settingNames.includes(rightSetting)) {
            continue;
          }
          steps.push({
            split: bundle.id,
            scene_id: sceneId,
            left_setting: leftSetting,
            right_setting: rightSetting,
            focus: safeString(preset.strategy.focus) || safeString(preset.description) || "",
            talking_points: [],
          });
        }
      }
    }
  }

  if (!steps.length && bundle.sceneOrder.length) {
    const sceneId = bundle.sceneOrder[0];
    const sceneObj = bundle.sceneById.get(sceneId);
    const leftSetting = sceneObj.settingNames.includes("calibration_v0")
      ? "calibration_v0"
      : sceneObj.settingNames[0];
    const rightSetting = sceneObj.settingNames.includes("calibration_v1")
      ? "calibration_v1"
      : sceneObj.settingNames[Math.min(1, sceneObj.settingNames.length - 1)];
    if (leftSetting && rightSetting) {
      steps.push({
        split: bundle.id,
        scene_id: sceneId,
        left_setting: leftSetting,
        right_setting: rightSetting,
        focus: "Fallback autoplay step",
        talking_points: [
          "Preset did not yield valid scene-setting steps.",
          "Using first available scene and settings.",
        ],
      });
    }
  }

  return steps;
}

function applyQueryOverrides() {
  const params = new URLSearchParams(window.location.search);

  const split = params.get("split");
  if (split && state.splitBundles.has(split)) {
    const bundle = state.splitBundles.get(split);
    if (bundle && bundle.available) {
      state.splitId = split;
      el.splitSelect.value = split;
      syncSceneOptions();
      syncSettingOptions();
    }
  }

  const scene = params.get("scene");
  const sceneObj = getCurrentBundle()?.sceneById.get(scene);
  if (scene && sceneObj) {
    state.sceneId = scene;
    el.sceneSelect.value = scene;
    syncSettingOptions();
  }

  const left = params.get("left");
  if (left && getCurrentScene()?.settingNames.includes(left)) {
    state.leftSetting = left;
    el.leftSettingSelect.value = left;
  }

  const right = params.get("right");
  if (right && getCurrentScene()?.settingNames.includes(right)) {
    state.rightSetting = right;
    el.rightSettingSelect.value = right;
  }

  const preset = params.get("preset");
  if (preset && state.presets.some((item) => item.id === preset)) {
    el.presetSelect.value = preset;
    compileSelectedPreset();
  }

  const stepSec = Number(params.get("stepSec"));
  if (Number.isFinite(stepSec) && stepSec >= 2 && stepSec <= 90) {
    el.stepDurationInput.value = String(stepSec);
  }

  const loopParam = params.get("loop");
  if (loopParam === "0" || loopParam === "false") {
    el.loopCheckbox.checked = false;
  }

  if (params.get("autoplay") === "1" || params.get("autoplay") === "true") {
    window.setTimeout(() => startAutoplay(), 120);
  }
}

function getSelectedPreset() {
  const selectedId = el.presetSelect.value;
  return state.presets.find((preset) => preset.id === selectedId) || null;
}

function startAutoplay() {
  if (!state.compiledSteps.length) {
    return;
  }
  stopAutoplay();
  state.autoplay.running = true;
  el.autoplayToggleBtn.textContent = "Stop Autoplay";

  const durationMs = Math.max(2, Number(el.stepDurationInput.value) || 8) * 1000;

  const runStep = () => {
    if (!state.autoplay.running) {
      return;
    }
    if (!state.compiledSteps.length) {
      stopAutoplay();
      return;
    }

    if (state.autoplay.stepIndex >= state.compiledSteps.length) {
      if (el.loopCheckbox.checked) {
        state.autoplay.stepIndex = 0;
      } else {
        stopAutoplay();
        return;
      }
    }

    applyAutoplayStep(state.autoplay.stepIndex);
    state.autoplay.stepIndex += 1;
    updateAutoplayStatus();
    state.autoplay.timer = window.setTimeout(runStep, durationMs);
  };

  runStep();
}

function stopAutoplay() {
  state.autoplay.running = false;
  if (state.autoplay.timer) {
    window.clearTimeout(state.autoplay.timer);
    state.autoplay.timer = null;
  }
  el.autoplayToggleBtn.textContent = "Start Autoplay";
  updateAutoplayStatus();
}

function applyAutoplayStep(stepIndex) {
  const step = state.compiledSteps[stepIndex];
  if (!step) {
    return;
  }

  if (state.splitId !== step.split) {
    state.splitId = step.split;
    el.splitSelect.value = step.split;
    syncSceneOptions();
  }

  if (state.sceneId !== step.scene_id) {
    state.sceneId = step.scene_id;
    el.sceneSelect.value = step.scene_id;
  }

  syncSettingOptions();
  state.leftSetting = step.left_setting;
  state.rightSetting = step.right_setting;
  el.leftSettingSelect.value = state.leftSetting;
  el.rightSettingSelect.value = state.rightSetting;

  refreshRender(step);
}

function updateAutoplayStatus() {
  const total = state.compiledSteps.length;
  const current = state.autoplay.running
    ? Math.min(state.autoplay.stepIndex + 1, Math.max(1, total))
    : Math.min(state.autoplay.stepIndex, Math.max(0, total));
  el.stepCounter.textContent = `Step ${total ? current : 0} / ${total}`;

  if (!total) {
    el.stepDescription.textContent = "No autoplay steps available";
    return;
  }

  const index = Math.min(Math.max(state.autoplay.stepIndex, 0), total - 1);
  const step = state.compiledSteps[index];
  if (!step) {
    el.stepDescription.textContent = state.autoplay.running ? "Autoplay running" : "Manual mode";
    return;
  }

  const focus = safeString(step.focus);
  el.stepDescription.textContent = focus
    ? `${step.scene_id}: ${SETTING_LABELS[step.left_setting] || step.left_setting} vs ${SETTING_LABELS[step.right_setting] || step.right_setting} - ${focus}`
    : `${step.scene_id}: ${step.left_setting} vs ${step.right_setting}`;
}

function refreshRender(activeStep = null) {
  const bundle = getCurrentBundle();
  const sceneObj = getCurrentScene();

  state.statusMessages = [];
  if (!bundle || !bundle.available) {
    state.statusMessages.push("Selected split has no evaluation_report.json yet.");
    renderEmptyState();
    setStatusBanner(state.statusMessages);
    return;
  }

  if (bundle.warnings.length) {
    state.statusMessages.push(...bundle.warnings);
  }
  if (!sceneObj) {
    state.statusMessages.push("No scene available for this split.");
    renderEmptyState();
    setStatusBanner(state.statusMessages);
    return;
  }

  const leftView = buildSettingView(bundle, sceneObj, state.leftSetting);
  const rightView = buildSettingView(bundle, sceneObj, state.rightSetting);

  renderPanel(panel.left, leftView, sceneObj);
  renderPanel(panel.right, rightView, sceneObj);
  renderDiffPanel(leftView, rightView, activeStep);
  setStatusBanner(state.statusMessages);
  updateAutoplayStatus();
}

function renderEmptyState() {
  const placeholder = {
    sceneId: "-",
    settingName: "-",
    displaySettingName: "Unavailable",
    status: "missing",
    sceneSummaryText: "No data available for this split.",
    relationText: "",
    objects: [],
    relations: [],
    relationHints: [],
    relationHintCount: 0,
    relationEvidenceLevel: "none",
    objectCount: 0,
    relationCount: 0,
    objectLabels: [],
    relationPredicates: [],
    qaExamples: [],
    groundingExamples: [],
    provenance: {},
    notes: [],
    failures: [],
    metrics: {},
  };
  renderPanel(panel.left, placeholder, null);
  renderPanel(panel.right, placeholder, null);
  renderDiffPanel(placeholder, placeholder, null);
}

function buildSettingView(bundle, sceneObj, settingName) {
  const settingEntry = sceneObj.settingsMap.get(settingName) || {};
  const metadata = settingEntry.metadata && typeof settingEntry.metadata === "object"
    ? settingEntry.metadata
    : {};
  const metrics = settingEntry.metrics && typeof settingEntry.metrics === "object"
    ? settingEntry.metrics
    : {};

  const languageKey = makeSceneSettingKey(sceneObj.sceneId, settingName);
  const languageRow = bundle.language.sceneByKey.get(languageKey) || null;
  const qaRows = bundle.language.qaByKey.get(languageKey) || [];
  const groundingRows = bundle.language.groundingByKey.get(languageKey) || [];

  const predictionResolution = sceneObj.predictionCache.get(settingName) || {
    payload: null,
    source: "summary_fallback",
    trace: ["summary_fallback"],
  };

  const predictionSummary = firstNonNullObject([
    metadata.prediction_summary_pre_repair,
    getPredictionSummaryFromDelta(sceneObj.deltaRow, settingName),
    languageRow && languageRow.metadata && languageRow.metadata.prediction_summary_pre_repair,
  ]);

  const structuredPrediction = predictionResolution.payload;
  const explicitObjects = structuredPrediction ? normalizePredictionObjects(structuredPrediction.objects) : [];
  const explicitRelations = structuredPrediction ? normalizePredictionRelations(structuredPrediction.relations) : [];

  const relationHints = resolveRelationHints({
    languageRow,
    structuredPrediction,
    predictionSummary,
  });

  const objectSummary = resolveObjectSummary({
    explicitObjects,
    predictionSummary,
    languageRow,
  });

  const relationSummary = resolveRelationSummary({
    explicitRelations,
    relationHints,
    predictionSummary,
  });

  const sceneSummaryText =
    safeString(languageRow && languageRow.scene_summary_text) ||
    buildFallbackSceneSummary(sceneObj.sceneId, settingName, objectSummary, relationSummary);

  const relationText =
    safeString(languageRow && languageRow.relation_text) ||
    buildFallbackRelationText(relationSummary);

  const qaExamples =
    collectQaExamples(languageRow, qaRows, objectSummary, relationSummary).slice(0, QA_LIST_LIMIT);

  const groundingExamples = collectGroundingExamples(
    languageRow,
    groundingRows,
    explicitObjects,
    objectSummary,
    relationSummary
  ).slice(0, GROUNDING_LIST_LIMIT);

  const relationPredicates = relationSummary.explicitPredicates.length
    ? relationSummary.explicitPredicates
    : relationSummary.hintPredicates;

  const relationEvidenceLevel = relationSummary.explicitRelations.length
    ? "explicit"
    : relationSummary.hintPredicates.length || relationSummary.hintCount > 0
      ? "hinted"
      : "none";

  const status = safeString(settingEntry.status) || "missing";

  const sourceClass =
    safeString(languageRow && languageRow.prediction_source_class) ||
    safeString(languageRow && languageRow.metadata && languageRow.metadata.prediction_source_class) ||
    (structuredPrediction && explicitRelations.length
      ? "explicit_structured_prediction"
      : structuredPrediction
        ? "structured_prediction_with_hint_only"
        : "summary_reconstructed");

  const provenance = {
    prediction_source_class: sourceClass,
    prediction_source_selected_from:
      safeString(languageRow && languageRow.metadata && languageRow.metadata.prediction_source_selected_from) ||
      predictionResolution.source,
    prediction_source_trace:
      (languageRow && languageRow.metadata && Array.isArray(languageRow.metadata.prediction_source_trace)
        ? languageRow.metadata.prediction_source_trace
        : predictionResolution.trace) || [],
    prediction_evidence_level:
      safeString(languageRow && languageRow.metadata && languageRow.metadata.prediction_evidence_level) ||
      relationEvidenceLevel,
    language_export_available: Boolean(languageRow),
    structured_prediction_available: Boolean(structuredPrediction),
    summary_fallback_used: !structuredPrediction,
    calibration_method: safeString(metadata.calibration_method) || null,
    calibration_execution: metadata.calibration_execution || null,
    generator_mode: safeString(metadata.generator_mode) || null,
    generator_name: safeString(metadata.generator_name) || null,
    setting_status: status,
    scene_tags: sceneObj.tags,
  };

  return {
    sceneId: sceneObj.sceneId,
    settingName,
    displaySettingName: SETTING_LABELS[settingName] || settingName,
    status,
    sceneSummaryText,
    relationText,
    objectCount: objectSummary.totalCount,
    relationCount: relationSummary.explicitRelations.length,
    relationHintCount: relationSummary.hintCount,
    relationHints: relationSummary.hintPredicates,
    relationEvidenceLevel,
    objectLabels: objectSummary.labels,
    relationPredicates,
    objects: objectSummary.objectsForView,
    relations: relationSummary.explicitRelations,
    qaExamples,
    groundingExamples,
    provenance,
    notes: Array.isArray(settingEntry.notes) ? settingEntry.notes : [],
    failures: Array.isArray(settingEntry.failures) ? settingEntry.failures : [],
    metrics,
    calibratedInputSummary: resolveCalibratedInputSummary(sceneObj.deltaRow, settingName, metadata),
    sourceType: sceneObj.sourceType,
  };
}

function resolveCalibratedInputSummary(deltaRow, settingName, metadata) {
  const alias = DELTA_ALIAS[settingName];
  if (alias && deltaRow && typeof deltaRow === "object") {
    const key = `calibrated_input_summary_${alias}`;
    if (deltaRow[key] && typeof deltaRow[key] === "object") {
      return deltaRow[key];
    }
  }
  if (metadata.calibrated_input_summary && typeof metadata.calibrated_input_summary === "object") {
    return metadata.calibrated_input_summary;
  }
  return null;
}

function resolveObjectSummary({ explicitObjects, predictionSummary, languageRow }) {
  if (explicitObjects.length) {
    const labels = sortedUnique(explicitObjects.map((obj) => obj.label));
    return {
      totalCount: explicitObjects.length,
      labels,
      objectsForView: explicitObjects,
      isSummaryFallback: false,
    };
  }

  const summaryLabels =
    sortedUnique(
      Array.isArray(predictionSummary && predictionSummary.object_labels)
        ? predictionSummary.object_labels.map((item) => safeString(item)).filter(Boolean)
        : []
    ) || [];

  const summaryCountRaw = Number(predictionSummary && predictionSummary.object_count);
  const summaryCount = Number.isFinite(summaryCountRaw)
    ? Math.max(summaryLabels.length, Math.max(0, Math.floor(summaryCountRaw)))
    : summaryLabels.length;

  const fallbackObjects = [];
  let syntheticIndex = 1;
  for (const label of summaryLabels) {
    fallbackObjects.push({
      object_id: `summary_${label}_${syntheticIndex}`,
      label,
      position: { x: 0, y: 0, z: 0 },
      size: { x: 1, y: 1, z: 1 },
      confidence: 0.0,
      attributes: {
        summary_reconstructed: true,
        geometry_unavailable: true,
      },
      count_hint: 1,
    });
    syntheticIndex += 1;
  }

  const missingCount = Math.max(0, summaryCount - fallbackObjects.length);
  if (missingCount > 0) {
    fallbackObjects.push({
      object_id: `summary_unknown_${syntheticIndex}`,
      label: "other_objects",
      position: { x: 0, y: 0, z: 0 },
      size: { x: 1, y: 1, z: 1 },
      confidence: 0.0,
      attributes: {
        summary_reconstructed: true,
        geometry_unavailable: true,
      },
      count_hint: missingCount,
    });
  }

  if (!fallbackObjects.length && languageRow) {
    fallbackObjects.push({
      object_id: "summary_unknown_1",
      label: "unknown",
      position: { x: 0, y: 0, z: 0 },
      size: { x: 1, y: 1, z: 1 },
      confidence: 0.0,
      attributes: {
        summary_reconstructed: true,
        geometry_unavailable: true,
      },
      count_hint: 0,
    });
  }

  return {
    totalCount: summaryCount,
    labels: summaryLabels,
    objectsForView: fallbackObjects,
    isSummaryFallback: true,
  };
}

function resolveRelationSummary({ explicitRelations, relationHints, predictionSummary }) {
  const explicitPredicates = sortedUnique(explicitRelations.map((rel) => rel.predicate));

  let hintPredicates = relationHints.predicates;
  if (!hintPredicates.length && predictionSummary && Array.isArray(predictionSummary.relation_predicates)) {
    hintPredicates = sortedUnique(predictionSummary.relation_predicates.map((item) => safeString(item)).filter(Boolean));
  }

  let hintCount = relationHints.count;
  if (hintCount <= 0 && predictionSummary && Number.isFinite(Number(predictionSummary.relation_count))) {
    hintCount = Math.max(0, Math.floor(Number(predictionSummary.relation_count)));
  }

  return {
    explicitRelations,
    explicitPredicates,
    hintPredicates,
    hintCount,
  };
}

function resolveRelationHints({ languageRow, structuredPrediction, predictionSummary }) {
  if (languageRow && Array.isArray(languageRow.relation_hint_predicates)) {
    return {
      predicates: sortedUnique(languageRow.relation_hint_predicates.map((item) => safeString(item)).filter(Boolean)),
      count: Number.isFinite(Number(languageRow.relation_hint_count))
        ? Math.max(0, Math.floor(Number(languageRow.relation_hint_count)))
        : 0,
    };
  }

  const metadata =
    structuredPrediction && structuredPrediction.metadata && typeof structuredPrediction.metadata === "object"
      ? structuredPrediction.metadata
      : {};

  const metadataPredicates = Array.isArray(metadata.relation_predicates)
    ? metadata.relation_predicates.map((item) => safeString(item)).filter(Boolean)
    : [];

  if (metadataPredicates.length) {
    return {
      predicates: sortedUnique(metadataPredicates),
      count: Number.isFinite(Number(metadata.relation_count_hint))
        ? Math.max(0, Math.floor(Number(metadata.relation_count_hint)))
        : metadataPredicates.length,
    };
  }

  const summaryPredicates = Array.isArray(predictionSummary && predictionSummary.relation_predicates)
    ? predictionSummary.relation_predicates.map((item) => safeString(item)).filter(Boolean)
    : [];
  return {
    predicates: sortedUnique(summaryPredicates),
    count: Number.isFinite(Number(predictionSummary && predictionSummary.relation_count))
      ? Math.max(0, Math.floor(Number(predictionSummary.relation_count)))
      : summaryPredicates.length,
  };
}

function collectQaExamples(languageRow, qaRows, objectSummary, relationSummary) {
  const fromSceneRow = languageRow && Array.isArray(languageRow.qa_examples) ? languageRow.qa_examples : [];
  if (fromSceneRow.length) {
    return fromSceneRow.map((item) => ({
      question: safeString(item.question),
      answer: safeString(item.answer),
      task_type: safeString(item.task_type) || "qa",
    }));
  }

  if (qaRows.length) {
    return qaRows.map((row) => ({
      question: safeString(row.question),
      answer: safeString(row.answer),
      task_type: safeString(row.task_type) || "qa",
    }));
  }

  const labelsText = objectSummary.labels.length ? objectSummary.labels.join(", ") : "none";
  const relationEvidence = relationSummary.explicitPredicates.length
    ? relationSummary.explicitPredicates.join(", ")
    : relationSummary.hintPredicates.length
      ? relationSummary.hintPredicates.join(", ")
      : "none";

  return [
    {
      task_type: "list_object_labels",
      question: "Which object labels appear in this scene-setting output?",
      answer: labelsText,
    },
    {
      task_type: "count_objects",
      question: "How many objects are represented?",
      answer: String(objectSummary.totalCount),
    },
    {
      task_type: "relation_predicates_list",
      question: "Which relation predicates appear in the available evidence?",
      answer: relationEvidence,
    },
  ];
}

function collectGroundingExamples(languageRow, groundingRows, explicitObjects, objectSummary, relationSummary) {
  const fromSceneRow = languageRow && Array.isArray(languageRow.grounding_examples)
    ? languageRow.grounding_examples
    : [];
  if (fromSceneRow.length) {
    return fromSceneRow.map((item) => ({
      text: safeString(item.text),
      task_type: safeString(item.task_type) || "grounding",
    }));
  }

  if (groundingRows.length) {
    return groundingRows.map((row) => ({
      text: safeString(row.text),
      task_type: safeString(row.task_type) || "grounding",
    }));
  }

  const examples = [];
  const geometryAvailable = hasMeaningfulGeometry(explicitObjects);
  if (geometryAvailable) {
    for (const obj of explicitObjects.slice(0, 3)) {
      examples.push({
        task_type: "referential_statement_geometric",
        text: `Object ${obj.object_id} (${obj.label}) appears near (${fmt(obj.position.x)}, ${fmt(obj.position.y)}, ${fmt(obj.position.z)}).`,
      });
    }
  } else {
    for (const obj of objectSummary.objectsForView.slice(0, 3)) {
      examples.push({
        task_type: "referential_statement_presence",
        text: `An object with label ${obj.label} is present in the exported structure.`,
      });
    }
  }

  if (!relationSummary.explicitRelations.length && relationSummary.hintPredicates.length) {
    examples.push({
      task_type: "relation_evidence_statement",
      text: `Hinted relation evidence is available for: ${relationSummary.hintPredicates.join(", ")}.`,
    });
  }

  if (!examples.length) {
    examples.push({
      task_type: "grounding",
      text: "No grounding evidence was exported for this scene-setting pair.",
    });
  }

  return examples;
}

function normalizePredictionObjects(objects) {
  if (!Array.isArray(objects)) {
    return [];
  }

  return objects
    .filter((obj) => obj && typeof obj === "object")
    .map((obj, index) => {
      const attributes = obj.attributes && typeof obj.attributes === "object" ? obj.attributes : {};
      return {
        object_id: safeString(obj.object_id) || `obj_${index + 1}`,
        label: safeString(obj.label) || "unknown",
        position: normalizePoint(obj.position),
        size: normalizePoint(obj.size, 1),
        confidence: Number.isFinite(Number(obj.confidence)) ? Number(obj.confidence) : 0,
        attributes,
        count_hint: Number.isFinite(Number(attributes.count_hint))
          ? Math.max(1, Math.floor(Number(attributes.count_hint)))
          : 1,
      };
    });
}

function normalizePredictionRelations(relations) {
  if (!Array.isArray(relations)) {
    return [];
  }

  return relations
    .filter((rel) => rel && typeof rel === "object")
    .map((rel) => ({
      subject_id: safeString(rel.subject_id) || "unknown_subject",
      predicate: safeString(rel.predicate) || "related-to",
      object_id: safeString(rel.object_id) || "unknown_object",
      score: Number.isFinite(Number(rel.score)) ? Number(rel.score) : 0,
      metadata: rel.metadata && typeof rel.metadata === "object" ? rel.metadata : {},
    }));
}

function normalizePoint(value, fallback = 0) {
  if (!value || typeof value !== "object") {
    return { x: fallback, y: fallback, z: fallback };
  }
  return {
    x: Number.isFinite(Number(value.x)) ? Number(value.x) : fallback,
    y: Number.isFinite(Number(value.y)) ? Number(value.y) : fallback,
    z: Number.isFinite(Number(value.z)) ? Number(value.z) : fallback,
  };
}

function getPredictionSummaryFromDelta(deltaRow, settingName) {
  if (!deltaRow || typeof deltaRow !== "object") {
    return null;
  }
  const alias = DELTA_ALIAS[settingName];
  if (!alias) {
    return null;
  }
  const summary = deltaRow[`prediction_summary_${alias}`];
  return summary && typeof summary === "object" ? summary : null;
}

function buildFallbackSceneSummary(sceneId, settingName, objectSummary, relationSummary) {
  const labels = objectSummary.labels.length ? objectSummary.labels.join(", ") : "none";
  const relationSegment = relationSummary.explicitRelations.length
    ? `${relationSummary.explicitRelations.length} explicit relations`
    : relationSummary.hintCount > 0 || relationSummary.hintPredicates.length
      ? `0 explicit relations, ${relationSummary.hintCount || relationSummary.hintPredicates.length} hinted`
      : "0 relation evidence";

  return `Scene ${sceneId} (${SETTING_LABELS[settingName] || settingName}) contains ${objectSummary.totalCount} objects [${labels}] and ${relationSegment}.`;
}

function buildFallbackRelationText(relationSummary) {
  if (relationSummary.explicitRelations.length) {
    return `Explicit relations: ${relationSummary.explicitRelations
      .slice(0, 12)
      .map((rel) => `${rel.subject_id} ${rel.predicate} ${rel.object_id}`)
      .join("; ")}`;
  }

  if (relationSummary.hintPredicates.length) {
    const hintCount = relationSummary.hintCount || relationSummary.hintPredicates.length;
    return `No explicit relation tuples were exported. Hinted relation evidence count=${hintCount}; predicates=${relationSummary.hintPredicates.join(", ")}.`;
  }

  return "No explicit or hinted relation evidence was exported.";
}

function renderPanel(panelRefs, view, sceneObj) {
  panelRefs.title.textContent = `${view.displaySettingName || view.settingName} - ${view.sceneId}`;
  panelRefs.sceneSummary.textContent = view.sceneSummaryText || "";
  panelRefs.relationText.textContent = view.relationText || "";

  setChip(panelRefs.evidenceChip, `evidence: ${view.relationEvidenceLevel}`, relationEvidenceChipClass(view));
  setChip(
    panelRefs.sourceChip,
    `source: ${safeString(view.provenance && view.provenance.prediction_source_class) || "unknown"}`,
    sourceChipClass(view)
  );
  setChip(panelRefs.statusChip, `status: ${view.status || "unknown"}`, statusChipClass(view.status));

  renderObjectsList(panelRefs.objectsList, view);
  renderRelationsList(panelRefs.relationsList, view);
  renderTextList(panelRefs.qaList, view.qaExamples.map((item) => `${item.question} -> ${item.answer || ""}`));
  renderTextList(panelRefs.groundingList, view.groundingExamples.map((item) => item.text || ""));

  panelRefs.provenance.textContent = JSON.stringify(
    {
      ...view.provenance,
      source_type: view.sourceType,
      calibrated_input_summary: view.calibratedInputSummary,
      metric_snapshot: pickMetricSnapshot(view.metrics),
      failures: view.failures,
      notes: view.notes,
    },
    null,
    2
  );

  renderSceneCanvas(panelRefs.canvas, view);

  if (sceneObj && sceneObj.tags.length) {
    state.statusMessages.push(`Scene ${sceneObj.sceneId} tags: ${sceneObj.tags.join(", ")}`);
  }
}

function renderObjectsList(target, view) {
  const lines = [];
  const objects = view.objects || [];
  if (!objects.length) {
    lines.push("No objects exported.");
  } else {
    for (const obj of objects.slice(0, 24)) {
      const countHint = Number.isFinite(Number(obj.count_hint)) ? Number(obj.count_hint) : 1;
      const isApprox = !!(obj.attributes && (obj.attributes.summary_reconstructed || obj.attributes.geometry_unavailable));
      if (isApprox) {
        lines.push(
          `${obj.object_id} (${obj.label}) [summary-level, geometry unavailable${countHint > 1 ? `, count_hint=${countHint}` : ""}]`
        );
      } else {
        lines.push(
          `${obj.object_id} (${obj.label}) pos=(${fmt(obj.position.x)}, ${fmt(obj.position.y)}, ${fmt(obj.position.z)}) conf=${fmt(obj.confidence)}`
        );
      }
    }
    if (objects.length > 24) {
      lines.push(`... +${objects.length - 24} more objects omitted from list`);
    }
  }

  renderTextList(target, lines);
}

function renderRelationsList(target, view) {
  const lines = [];
  if (view.relations && view.relations.length) {
    for (const rel of view.relations.slice(0, 24)) {
      lines.push(`${rel.subject_id} ${rel.predicate} ${rel.object_id} (score=${fmt(rel.score)})`);
    }
    if (view.relations.length > 24) {
      lines.push(`... +${view.relations.length - 24} more relations omitted from list`);
    }
  } else if (view.relationHints && view.relationHints.length) {
    const hintCount = view.relationHintCount || view.relationHints.length;
    lines.push(`No explicit tuples. Hinted predicates: ${view.relationHints.join(", ")} (count_hint=${hintCount})`);
  } else {
    lines.push("No relation evidence exported.");
  }
  renderTextList(target, lines);
}

function renderTextList(target, lines) {
  clearChildren(target);
  const normalized = lines.filter((line) => safeString(line));
  if (!normalized.length) {
    const li = document.createElement("li");
    li.textContent = "No entries";
    target.appendChild(li);
    return;
  }
  for (const line of normalized) {
    const li = document.createElement("li");
    li.textContent = line;
    target.appendChild(li);
  }
}

function renderSceneCanvas(canvas, view) {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawCanvasBackground(ctx, canvas.width, canvas.height);

  const renderBundle = buildRenderableObjects(view.objects || [], canvas.width, canvas.height);
  const objectById = new Map(renderBundle.nodes.map((node) => [node.object_id, node]));

  drawRelations(ctx, view.relations || [], objectById);
  drawNodes(ctx, renderBundle.nodes);

  if (!view.relations.length && view.relationHints.length) {
    const hintText = `Hinted predicates: ${view.relationHints.join(", ")}`;
    drawHintBanner(ctx, canvas.width, hintText);
  }

  if (renderBundle.omittedCount > 0) {
    drawHintBanner(ctx, canvas.width, `Visualization clipped: +${renderBundle.omittedCount} objects not drawn`, 42);
  }
}

function drawCanvasBackground(ctx, width, height) {
  const bg = ctx.createLinearGradient(0, 0, 0, height);
  bg.addColorStop(0, "rgba(13, 27, 40, 0.95)");
  bg.addColorStop(1, "rgba(7, 13, 22, 0.98)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(130, 156, 186, 0.18)";
  ctx.lineWidth = 1;
  for (let x = 20; x < width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 20; y < height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function buildRenderableObjects(objects, width, height) {
  const normalizedObjects = objects.slice(0, CANVAS_MAX_OBJECTS);
  const omittedCount = Math.max(0, objects.length - normalizedObjects.length);
  const meaningfulGeometry = hasMeaningfulGeometry(normalizedObjects);

  let nodes = [];
  if (meaningfulGeometry) {
    nodes = layoutObjectsFromGeometry(normalizedObjects, width, height);
  } else {
    nodes = layoutObjectsInCircle(normalizedObjects, width, height);
  }

  return {
    nodes,
    omittedCount,
    meaningfulGeometry,
  };
}

function hasMeaningfulGeometry(objects) {
  if (!Array.isArray(objects) || objects.length < 2) {
    return false;
  }

  const usable = objects.filter((obj) => {
    const attrs = obj.attributes || {};
    if (attrs.summary_reconstructed || attrs.geometry_unavailable) {
      return false;
    }
    return [obj.position.x, obj.position.y, obj.position.z].every((value) => Number.isFinite(Number(value)));
  });

  if (usable.length < 2) {
    return false;
  }

  const xs = usable.map((obj) => obj.position.x);
  const zs = usable.map((obj) => obj.position.z);
  const ys = usable.map((obj) => obj.position.y);

  const spanX = Math.max(...xs) - Math.min(...xs);
  const spanZ = Math.max(...zs) - Math.min(...zs);
  const spanY = Math.max(...ys) - Math.min(...ys);

  return spanX > 0.05 || spanZ > 0.05 || spanY > 0.05;
}

function layoutObjectsFromGeometry(objects, width, height) {
  const margin = 28;
  const xValues = objects.map((obj) => obj.position.x);
  const zValues = objects.map((obj) => obj.position.z);
  const yValues = objects.map((obj) => obj.position.y);

  const useZ = span(zValues) > span(yValues) * 0.8;
  const verticalValues = useZ ? zValues : yValues;

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...verticalValues);
  const maxY = Math.max(...verticalValues);

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);

  return objects.map((obj) => {
    const nx = (obj.position.x - minX) / spanX;
    const ny = ((useZ ? obj.position.z : obj.position.y) - minY) / spanY;

    const px = margin + nx * (width - margin * 2);
    const py = height - margin - ny * (height - margin * 2);

    const sizeScale = Math.max(obj.size.x, obj.size.y, obj.size.z, 0.2);
    const w = clamp(28 + sizeScale * 12, 26, 72);
    const h = clamp(20 + sizeScale * 9, 18, 54);

    return {
      ...obj,
      renderX: px,
      renderY: py,
      renderW: w,
      renderH: h,
      approx: false,
    };
  });
}

function layoutObjectsInCircle(objects, width, height) {
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.33;

  if (objects.length === 1) {
    const obj = objects[0];
    return [
      {
        ...obj,
        renderX: centerX,
        renderY: centerY,
        renderW: 76,
        renderH: 34,
        approx: true,
      },
    ];
  }

  return objects.map((obj, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, objects.length);
    const px = centerX + radius * Math.cos(angle);
    const py = centerY + radius * Math.sin(angle);
    return {
      ...obj,
      renderX: px,
      renderY: py,
      renderW: 78,
      renderH: 34,
      approx: true,
    };
  });
}

function drawRelations(ctx, relations, objectById) {
  if (!relations.length) {
    return;
  }

  for (const rel of relations.slice(0, 60)) {
    const source = objectById.get(rel.subject_id);
    const target = objectById.get(rel.object_id);
    if (!source || !target) {
      continue;
    }

    ctx.strokeStyle = colorForLabel(rel.predicate, 0.8);
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(source.renderX, source.renderY);
    ctx.lineTo(target.renderX, target.renderY);
    ctx.stroke();

    const midX = (source.renderX + target.renderX) / 2;
    const midY = (source.renderY + target.renderY) / 2;
    ctx.fillStyle = "rgba(224, 236, 248, 0.92)";
    ctx.font = "11px \"JetBrains Mono\", monospace";
    ctx.textAlign = "center";
    ctx.fillText(truncate(rel.predicate, 18), midX, midY - 4);
  }
}

function drawNodes(ctx, nodes) {
  for (const node of nodes) {
    const x = node.renderX - node.renderW / 2;
    const y = node.renderY - node.renderH / 2;

    const fill = colorForLabel(node.label, node.approx ? 0.26 : 0.5);
    const stroke = colorForLabel(node.label, node.approx ? 0.62 : 0.86);

    roundRect(ctx, x, y, node.renderW, node.renderH, 7);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    const countHint = Number.isFinite(Number(node.count_hint)) ? Number(node.count_hint) : 1;
    const labelText = countHint > 1 ? `${node.label} x${countHint}` : node.label;

    ctx.fillStyle = "#f3f8ff";
    ctx.textAlign = "center";
    ctx.font = node.approx
      ? "bold 11px \"Avenir Next\", sans-serif"
      : "bold 11px \"Avenir Next\", sans-serif";
    ctx.fillText(truncate(labelText, 20), node.renderX, node.renderY + 4);
  }
}

function drawHintBanner(ctx, width, text, top = 16) {
  const bannerWidth = width - 26;
  roundRect(ctx, 13, top, bannerWidth, 18, 8);
  ctx.fillStyle = "rgba(24, 37, 53, 0.86)";
  ctx.fill();
  ctx.strokeStyle = "rgba(111, 143, 177, 0.6)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = "#dce8f7";
  ctx.font = "11px \"JetBrains Mono\", monospace";
  ctx.textAlign = "left";
  ctx.fillText(truncate(text, 96), 20, top + 12.5);
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function renderDiffPanel(leftView, rightView, activeStep) {
  const leftLabels = new Set(leftView.objectLabels || []);
  const rightLabels = new Set(rightView.objectLabels || []);
  const leftPredicates = new Set(leftView.relationPredicates || []);
  const rightPredicates = new Set(rightView.relationPredicates || []);

  renderDiffChips(el.diffObjectLabels, leftLabels, rightLabels);
  renderDiffChips(el.diffRelationPredicates, leftPredicates, rightPredicates);

  clearChildren(el.diffCounts);
  addChip(el.diffCounts, `objects: ${leftView.objectCount} -> ${rightView.objectCount}`, "");
  addChip(el.diffCounts, `relations: ${leftView.relationCount} -> ${rightView.relationCount}`, "");
  addChip(el.diffCounts, `evidence: ${leftView.relationEvidenceLevel} -> ${rightView.relationEvidenceLevel}`, "");
  addChip(el.diffCounts, `status: ${leftView.status} -> ${rightView.status}`, "");

  const relDelta = numericDelta(leftView.metrics.calibration_reliability, rightView.metrics.calibration_reliability);
  if (relDelta !== null) {
    addChip(
      el.diffCounts,
      `calibration_reliability ==${fmt(relDelta)}`,
      relDelta >= 0 ? "good" : "warn"
    );
  }

  const violationDelta = numericDelta(
    leftView.metrics.structured_violation_count_before_repair,
    rightView.metrics.structured_violation_count_before_repair
  );
  if (violationDelta !== null) {
    addChip(
      el.diffCounts,
      `violation_before_repair ==${fmt(violationDelta)}`,
      violationDelta <= 0 ? "good" : "warn"
    );
  }

  const narrationLines = buildNarrationLines(leftView, rightView, activeStep);
  renderTextList(el.narrationList, narrationLines);
}

function renderDiffChips(target, leftSet, rightSet) {
  clearChildren(target);
  const onlyLeft = sortedUnique([...leftSet].filter((item) => !rightSet.has(item)));
  const onlyRight = sortedUnique([...rightSet].filter((item) => !leftSet.has(item)));
  const common = sortedUnique([...leftSet].filter((item) => rightSet.has(item)));

  if (!onlyLeft.length && !onlyRight.length && !common.length) {
    addChip(target, "no evidence", "warn");
    return;
  }

  for (const item of onlyLeft) {
    addChip(target, `- ${item}`, "diff-remove");
  }
  for (const item of onlyRight) {
    addChip(target, `+ ${item}`, "diff-add");
  }
  for (const item of common) {
    addChip(target, `= ${item}`, "");
  }
}

function buildNarrationLines(leftView, rightView, activeStep) {
  const lines = [];
  if (activeStep && activeStep.focus) {
    lines.push(`Focus: ${activeStep.focus}`);
  }
  if (activeStep && Array.isArray(activeStep.talking_points) && activeStep.talking_points.length) {
    lines.push(...activeStep.talking_points.map((item) => `Narration: ${item}`));
  }

  const leftLabelSet = new Set(leftView.objectLabels || []);
  const rightLabelSet = new Set(rightView.objectLabels || []);
  const addedLabels = sortedUnique([...rightLabelSet].filter((item) => !leftLabelSet.has(item)));
  const removedLabels = sortedUnique([...leftLabelSet].filter((item) => !rightLabelSet.has(item)));

  if (addedLabels.length || removedLabels.length) {
    lines.push(`Label delta -> added: ${addedLabels.join(", ") || "none"}; removed: ${removedLabels.join(", ") || "none"}.`);
  }

  const leftPredSet = new Set(leftView.relationPredicates || []);
  const rightPredSet = new Set(rightView.relationPredicates || []);
  const addedPreds = sortedUnique([...rightPredSet].filter((item) => !leftPredSet.has(item)));
  const removedPreds = sortedUnique([...leftPredSet].filter((item) => !rightPredSet.has(item)));
  if (addedPreds.length || removedPreds.length) {
    lines.push(`Relation predicate delta -> added: ${addedPreds.join(", ") || "none"}; removed: ${removedPreds.join(", ") || "none"}.`);
  }

  lines.push(`Relation evidence -> ${leftView.relationEvidenceLevel} vs ${rightView.relationEvidenceLevel}.`);
  lines.push(`Counts -> objects ${leftView.objectCount} vs ${rightView.objectCount}, relations ${leftView.relationCount} vs ${rightView.relationCount}.`);

  if (!lines.length) {
    lines.push("Manual mode: choose a preset or scene-setting pair to begin comparison.");
  }

  return lines;
}

function setStatusBanner(messages) {
  const uniqueMessages = sortedUnique(messages.filter((item) => safeString(item)));
  if (!uniqueMessages.length) {
    el.statusBanner.classList.add("hidden");
    el.statusBanner.textContent = "";
    return;
  }

  el.statusBanner.classList.remove("hidden");
  el.statusBanner.textContent = uniqueMessages.join(" | ");
}

function setChip(chipEl, text, className) {
  chipEl.textContent = text;
  chipEl.className = "chip";
  if (className) {
    chipEl.classList.add(className);
  }
}

function addChip(target, text, className) {
  const span = document.createElement("span");
  span.className = "chip";
  if (className) {
    span.classList.add(className);
  }
  span.textContent = text;
  target.appendChild(span);
}

function relationEvidenceChipClass(view) {
  if (view.relationEvidenceLevel === "explicit") {
    return "good";
  }
  if (view.relationEvidenceLevel === "hinted") {
    return "warn";
  }
  return "bad";
}

function sourceChipClass(view) {
  const source = safeString(view.provenance && view.provenance.prediction_source_class);
  if (source === "explicit_structured_prediction") {
    return "good";
  }
  if (source === "structured_prediction_with_hint_only") {
    return "warn";
  }
  return "bad";
}

function statusChipClass(status) {
  if (status === "success") {
    return "good";
  }
  if (status === "failed") {
    return "bad";
  }
  return "warn";
}

function pickMetricSnapshot(metrics) {
  return {
    calibration_reliability: numericOrNull(metrics.calibration_reliability),
    calibration_up_axis_error_deg: numericOrNull(metrics.calibration_up_axis_error_deg),
    calibration_horizontal_error_deg: numericOrNull(metrics.calibration_horizontal_error_deg),
    structured_violation_count_before_repair: numericOrNull(metrics.structured_violation_count_before_repair),
    actionable_relation_f1: numericOrNull(metrics.actionable_relation_f1),
  };
}

function getCurrentBundle() {
  return state.splitBundles.get(state.splitId) || null;
}

function getCurrentScene() {
  const bundle = getCurrentBundle();
  if (!bundle || !bundle.available || !state.sceneId) {
    return null;
  }
  return bundle.sceneById.get(state.sceneId) || null;
}

function orderedSettings(settingNames) {
  const known = [];
  const unknown = [];
  for (const name of settingNames) {
    if (SETTING_ORDER.includes(name)) {
      known.push(name);
    } else {
      unknown.push(name);
    }
  }
  known.sort((a, b) => SETTING_ORDER.indexOf(a) - SETTING_ORDER.indexOf(b));
  unknown.sort();
  return [...known, ...unknown];
}

function makeSceneSettingKey(sceneId, setting) {
  return `${sceneId}::${setting}`;
}

async function fetchJsonMaybe(url) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (_error) {
    return null;
  }
}

async function fetchJsonlMaybe(url) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return [];
    }
    const text = await response.text();
    const rows = [];
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        rows.push(JSON.parse(trimmed));
      } catch (_error) {
        // Ignore malformed line and keep demo resilient.
      }
    }
    return rows;
  } catch (_error) {
    return [];
  }
}

function clearChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function safeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function sortedUnique(values) {
  return Array.from(new Set(values)).sort((a, b) => String(a).localeCompare(String(b)));
}

function firstNonNullObject(candidates) {
  for (const candidate of candidates) {
    if (candidate && typeof candidate === "object") {
      return candidate;
    }
  }
  return null;
}

function span(values) {
  if (!values.length) {
    return 0;
  }
  return Math.max(...values) - Math.min(...values);
}

function truncate(text, maxLength) {
  const str = String(text || "");
  if (str.length <= maxLength) {
    return str;
  }
  return `${str.slice(0, Math.max(0, maxLength - 1))}...`;
}

function fmt(value) {
  if (!Number.isFinite(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function numericOrNull(value) {
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

function numericDelta(leftValue, rightValue) {
  const left = numericOrNull(leftValue);
  const right = numericOrNull(rightValue);
  if (left === null || right === null) {
    return null;
  }
  return right - left;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function colorForLabel(label, alpha = 1) {
  const str = safeString(label) || "object";
  let hash = 0;
  for (let index = 0; index < str.length; index += 1) {
    hash = (hash * 31 + str.charCodeAt(index)) | 0;
  }
  const hue = Math.abs(hash) % 360;
  return `hsla(${hue}, 62%, 56%, ${alpha})`;
}
