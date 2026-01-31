
// === Setup ===
const baseUrl = "http://127.0.0.1:8000";

const startBtn = document.getElementById("startBtn");
const modal = document.getElementById("questionnaireModal");
const closeBtn = document.querySelector(".close-button");
const questionnaireSection = document.getElementById("questionnaireSection");
const questionText = document.getElementById("questionText");
const optionsDiv = document.getElementById("options");
const nextBtn = document.getElementById("nextBtn");
const prevBtn = document.getElementById("prevBtn");
const resultsSection = document.getElementById("resultsSection");
const personalityResult = document.getElementById("personalityResult");
const recommendationResult = document.getElementById("recommendationResult");
const restartBtn = document.getElementById("restartBtn");
const progressBar = document.getElementById("progressBar");
const loadingIndicator = document.getElementById("loadingIndicator");
const resultsContent = document.getElementById("resultsContent");

let currentQuestionIndex = 0;
let stage = "personality";
let questions = [];
let answers = {};
let selectedDomain = "";
let domainQuestions = [];

const personalityQuestionList = [
  { trait: "Openness", question: "Do you enjoy trying new things?" },
  { trait: "Conscientiousness", question: "Are you a detail-oriented person?" },
  { trait: "Extraversion", question: "Do you gain energy from social interactions?" },
  { trait: "Agreeableness", question: "Do you prioritize harmony in relationships?" },
  { trait: "Neuroticism", question: "Do you often feel anxious or stressed?" }
];

const domainQuestionSets = {
  "Books": [
    { id: "Q1", question: "Do you enjoy books with magical elements?" },
    { id: "Q2", question: "Do you prefer fiction over non-fiction?" },
    { id: "Q3", question: "Do you like shorter or longer books?" }
  ],
  "Movies": [
    { id: "Q1", question: "Do you enjoy emotional dramas?" },
    { id: "Q2", question: "Do you prefer action over comedy?" },
    { id: "Q3", question: "Do you like watching thrillers?" }
  ],
  "Music": [
    { id: "Q1", question: "Do you enjoy relaxing music more than energetic?" },
    { id: "Q2", question: "Do you prefer instrumental over lyrical?" },
    { id: "Q3", question: "Do you often explore new genres?" }
  ],
  "Games": [
    { id: "Q1", question: "Do you enjoy story-driven games?" },
    { id: "Q2", question: "Do you prefer multiplayer over single-player?" },
    { id: "Q3", question: "Do you enjoy open-world exploration?" }
  ]
};

startBtn.addEventListener("click", () => {
  modal.style.display = "block";
  document.body.style.overflow = "hidden";
  startQuestionnaire();
});

closeBtn.addEventListener("click", () => {
  modal.style.display = "none";
  document.body.style.overflow = "auto";
  resetQuestionnaire();
});

window.addEventListener("click", function (event) {
  if (event.target === modal) {
    modal.style.display = "none";
    document.body.style.overflow = "auto";
    resetQuestionnaire();
  }
});

function startQuestionnaire() {
  questionnaireSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  stage = "personality";
  questions = personalityQuestionList;
  currentQuestionIndex = 0;
  showQuestion(currentQuestionIndex);
}

function resetQuestionnaire() {
  currentQuestionIndex = 0;
  answers = {};
  selectedDomain = "";
  domainQuestions = [];
  stage = "personality";
  questionnaireSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
}

function showQuestion(index) {
  optionsDiv.innerHTML = "";
  prevBtn.classList.toggle("hidden", index === 0);

  if (index >= questions.length) {
    if (stage === "personality") {
      askDomainSelection();
    } else {
      return getRecommendations();
    }
    return;
  }

  const question = questions[index];
  questionText.textContent = question.question;

  for (let i = 1; i <= 5; i++) {
    const btn = document.createElement("button");
    btn.textContent = i;
    btn.onclick = () => {
      answers[stage === "personality" ? question.trait : question.id] = i;
      setTimeout(() => {
        currentQuestionIndex++;
        showQuestion(currentQuestionIndex);
      }, 200);
    };

    if (answers[question.trait] === i || answers[question.id] === i) {
      btn.classList.add("selected");
    }

    optionsDiv.appendChild(btn);
  }
}

function askDomainSelection() {
  questionText.textContent = "Choose your preferred entertainment domain:";
  optionsDiv.innerHTML = "";
  const domains = ["Books", "Movies", "Music", "Games"];

  domains.forEach(domain => {
    const btn = document.createElement("button");
    btn.textContent = domain;
    btn.onclick = () => {
      selectedDomain = domain;
      answers["selected_domain"] = domain;
      stage = "domain";
      questions = domainQuestionSets[domain];
      currentQuestionIndex = 0;
      showQuestion(currentQuestionIndex);
    };
    optionsDiv.appendChild(btn);
  });
}

async function getRecommendations() {
  questionnaireSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  // Show loading, hide content
  loadingIndicator.classList.remove("hidden");
  resultsContent.classList.add("hidden");

  // Fallback if selectedDomain was lost
  if (!selectedDomain && answers.selected_domain) {
    selectedDomain = answers.selected_domain;
  }

  const general_scores = {};
  const domain_scores = {};

  Object.entries(answers).forEach(([key, value]) => {
    if (["Q1", "Q2", "Q3"].includes(key)) {
      domain_scores[key] = value;
    } else if (key !== "selected_domain") {
      general_scores[key] = value;
    }
  });

  if (!selectedDomain) {
    console.error("No domain selected");
    recommendationResult.textContent = "Please select a domain.";
    return;
  }

  try {
    const response = await fetch(`${baseUrl}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        domain: selectedDomain,
        general_scores: general_scores,
        domain_responses: domain_scores
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Server error: ${response.status} - ${errText}`);
    }

    const data = await response.json();

    if (!data || !data.personality_traits || !data.recommendations) {
      throw new Error("Incomplete data received from server.");
    }

    personalityResult.innerHTML = `
      <h3>Your Top Trait: ${data.dominant_trait}</h3>
      <p>${Object.entries(data.personality_traits)
        .map(([trait, score]) => `${trait}: ${score.toFixed(2)}`)
        .join("<br>")}</p>`;

    recommendationResult.innerHTML = `
      <h3>Recommended ${selectedDomain} (${data.recommended_genre}):</h3>
      ${data.recommendations.map(rec => `<div>🎯 ${rec}</div>`).join("")}
    `;

    // Hide loading, show content
    loadingIndicator.classList.add("hidden");
    resultsContent.classList.remove("hidden");

  } catch (error) {
    console.error("Recommendation error:", error);
    loadingIndicator.classList.add("hidden");
    resultsContent.classList.remove("hidden");
    recommendationResult.textContent = "Error getting recommendations. Please make sure the backend server (main.py) is running.";
  }
}


prevBtn.onclick = () => {
  if (currentQuestionIndex > 0) {
    currentQuestionIndex--;
    showQuestion(currentQuestionIndex);
  }
};

restartBtn.onclick = () => {
  resetQuestionnaire();
  startQuestionnaire();
};
