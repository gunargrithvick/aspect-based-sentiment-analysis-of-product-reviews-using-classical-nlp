"use strict";

const sentimentClasses = ["positive", "negative", "neutral", "conflict"];
let latestAnalysis = null;

function escapeHtml(value) {
    return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function highlightReview(text, results) {
    let output = "";
    let cursor = 0;
    [...results].sort((a, b) => a.start - b.start).forEach((result) => {
        if (result.start < cursor || result.end > text.length || result.start >= result.end) return;
        output += escapeHtml(text.slice(cursor, result.start));
        output += `<mark class="highlight-${escapeHtml(result.sentiment)}" title="${escapeHtml(result.sentiment)}">${escapeHtml(text.slice(result.start, result.end))}</mark>`;
        cursor = result.end;
    });
    return output + escapeHtml(text.slice(cursor));
}

function updateSummary(summary) {
    document.querySelector("#aspect-count").textContent = summary.aspect_count;
    sentimentClasses.forEach((sentiment) => {
        document.querySelector(`#${sentiment}-count`).textContent = summary.sentiment_counts[sentiment];
    });
}

function renderResults(data) {
    latestAnalysis = data;
    updateSummary(data.summary);
    document.querySelector("#highlighted-review").innerHTML = highlightReview(data.review, data.results);
    const tableBody = document.querySelector("#results-table");
    tableBody.innerHTML = "";
    if (data.results.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="3" class="empty-cell">No explicit product aspects were detected.</td></tr>';
    } else {
        data.results.forEach((result) => {
            const row = document.createElement("tr");
            row.innerHTML = `<td><strong>${escapeHtml(result.aspect)}</strong></td><td><span class="sentiment-pill sentiment-${escapeHtml(result.sentiment)}">${escapeHtml(result.sentiment)}</span></td><td>${result.start}–${result.end}</td>`;
            tableBody.appendChild(row);
        });
    }
    document.querySelector("#results-section").hidden = false;
    document.querySelector("#results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showMessage(message) {
    const element = document.querySelector("#message");
    element.textContent = message;
    element.hidden = false;
}

async function checkHealth() {
    const status = document.querySelector("#model-status");
    try {
        const response = await fetch("/health");
        const data = await response.json();
        status.textContent = data.models_loaded ? "Models ready" : "Models missing";
        status.classList.add(data.models_loaded ? "ready" : "error");
    } catch (error) {
        status.textContent = "Server unavailable";
        status.classList.add("error");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const textarea = document.querySelector("#review-text");
    const form = document.querySelector("#analysis-form");
    const button = document.querySelector("#analyze-button");
    document.querySelectorAll(".example-button").forEach((exampleButton) => {
        exampleButton.addEventListener("click", () => { textarea.value = exampleButton.dataset.example; textarea.focus(); });
    });
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        document.querySelector("#message").hidden = true;
        button.disabled = true;
        button.querySelector("span").textContent = "Analyzing...";
        try {
            const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: textarea.value }) });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "The review could not be analyzed.");
            renderResults(data);
        } catch (error) { showMessage(error.message); }
        finally { button.disabled = false; button.querySelector("span").textContent = "Analyze review"; }
    });
    document.querySelector("#download-button").addEventListener("click", () => {
        if (!latestAnalysis) return;
        const file = new Blob([JSON.stringify(latestAnalysis, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(file);
        const link = document.createElement("a");
        link.href = url; link.download = "absa_analysis_results.json"; link.click(); URL.revokeObjectURL(url);
    });
    checkHealth();
});
