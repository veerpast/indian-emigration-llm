const modelData = {
  mistral: {
    size: "4.37",
    name: "Mistral 7B Instruct v0.3",
    repo: "mistralai/Mistral-7B-Instruct-v0.3",
    rank: "64",
    alpha: "128",
    dropout: "0.05",
    recipe: "3 epochs · lr 1e-4*",
    note: "*Optimizer recipe is reported in the project log; no Mistral run_config.json is present."
  },
  granite: {
    size: "4.94",
    name: "Granite 3.3 8B Instruct",
    repo: "ibm-granite/granite-3.3-8b-instruct",
    rank: "64",
    alpha: "128",
    dropout: "0.05",
    recipe: "3 epochs · effective batch 32",
    note: "Recorded in run_config.json: lr 1e-4, max length 1024, cosine schedule, seed 20260926."
  },
  qwen: {
    size: "9.00",
    name: "Qwen3 14B",
    repo: "Qwen/Qwen3-14B",
    rank: "16",
    alpha: "32",
    dropout: "0.05",
    recipe: "1 epoch · effective batch 8",
    note: "Recorded in run_config.json: lr 2e-4, max length 1024, cosine schedule, thinking disabled."
  }
};

const modelBindings = {
  size: document.querySelector("#modelSize"),
  name: document.querySelector("#modelName"),
  repo: document.querySelector("#modelRepo"),
  rank: document.querySelector("#modelRank"),
  alpha: document.querySelector("#modelAlpha"),
  dropout: document.querySelector("#modelDropout"),
  recipe: document.querySelector("#modelRecipe"),
  note: document.querySelector("#modelNote")
};

document.querySelectorAll("[data-model]").forEach((tab) => {
  tab.addEventListener("click", () => {
    const selected = modelData[tab.dataset.model];
    document.querySelectorAll("[data-model]").forEach((item) => item.setAttribute("aria-selected", "false"));
    tab.setAttribute("aria-selected", "true");
    Object.entries(modelBindings).forEach(([key, element]) => { element.textContent = selected[key]; });
  });
});

const answers = {
  ecr: {
    topic: "Passport status",
    question: "What is the difference between ECR and ECNR?",
    text: "ECR and ECNR are passport emigration-status categories used in specific Indian emigration contexts. The correct status and any clearance requirement depend on the passport holder, destination, and current official rules. Verify the passport endorsement and consult the latest Passport Seva and Protector General of Emigrants guidance before travel."
  },
  interview: {
    topic: "Interview preparation",
    question: "How should a student explain their university choice?",
    text: "A strong answer is personal, specific, and consistent with the application: name the programme features, explain how they connect to prior study and career goals, and show that the applicant has compared realistic alternatives. Avoid memorised claims and never invent rankings, faculty links, or outcomes."
  },
  documents: {
    topic: "Financial evidence",
    question: "What financial documents may a student be asked for?",
    text: "Requirements differ by country, visa category, institution, and date. Common evidence can include official bank statements, sanctioned education-loan documents, scholarship letters, sponsor evidence, income records, and proof that funds are genuinely available. Use the current official checklist for the exact application—this example is not a substitute."
  }
};

const answerTopic = document.querySelector("#answerTopic");
const answerQuestion = document.querySelector("#answerQuestion");
const answerText = document.querySelector("#answerText");
const promptButtons = document.querySelectorAll("[data-prompt]");
promptButtons[0].classList.add("active");
promptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const answer = answers[button.dataset.prompt];
    promptButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    answerTopic.textContent = answer.topic;
    answerQuestion.textContent = answer.question;
    answerText.textContent = answer.text;
  });
});

const slides = [...document.querySelectorAll("[data-slide]")];
const presentButton = document.querySelector("#presentButton");
const controls = document.querySelector("#presentationControls");
const counter = document.querySelector("#slideCounter");
let currentSlide = 0;

function showSlide(index) {
  currentSlide = Math.max(0, Math.min(slides.length - 1, index));
  slides[currentSlide].scrollIntoView({ behavior: "smooth", block: "start" });
  counter.textContent = `${currentSlide + 1} / ${slides.length}`;
}

async function enterPresentation() {
  document.body.classList.add("presenting");
  controls.hidden = false;
  if (document.documentElement.requestFullscreen) {
    try { await document.documentElement.requestFullscreen(); } catch (_) { /* browser may deny */ }
  }
  showSlide(0);
}

async function exitPresentation() {
  document.body.classList.remove("presenting");
  controls.hidden = true;
  if (document.fullscreenElement) await document.exitFullscreen();
}

presentButton.addEventListener("click", enterPresentation);
document.querySelector("#previousSlide").addEventListener("click", () => showSlide(currentSlide - 1));
document.querySelector("#nextSlide").addEventListener("click", () => showSlide(currentSlide + 1));
document.querySelector("#exitPresentation").addEventListener("click", exitPresentation);

document.addEventListener("keydown", (event) => {
  if (!document.body.classList.contains("presenting")) return;
  if (["ArrowRight", "PageDown", " "].includes(event.key)) { event.preventDefault(); showSlide(currentSlide + 1); }
  if (["ArrowLeft", "PageUp"].includes(event.key)) { event.preventDefault(); showSlide(currentSlide - 1); }
  if (event.key === "Escape") exitPresentation();
});

document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && document.body.classList.contains("presenting")) {
    document.body.classList.remove("presenting");
    controls.hidden = true;
  }
});

