const glossaryItems = [
    {
        term: "Total Return",
        plain: "How much the strategy grew over the full test. If you started with 1.0, total return shows where you ended up.",
    },
    {
        term: "Sharpe Ratio",
        plain: "How much return the strategy generated for each unit of volatility. Higher is usually better.",
    },
    {
        term: "Max Drawdown",
        plain: "The worst drop from a previous peak. This is one of the easiest ways to judge how painful a strategy felt.",
    },
    {
        term: "Alpha and Beta",
        plain: "Beta shows how much the strategy moves with the benchmark. Alpha estimates return left over after that market exposure.",
    },
    {
        term: "Information Ratio",
        plain: "Measures how consistently the strategy beat the benchmark after adjusting for active risk.",
    },
];

const sortOptions = [
    { value: "momentum_3m", label: "3M Momentum" },
    { value: "avg_turnover", label: "Average Turnover" },
    { value: "momentum_1m", label: "1M Momentum" },
    { value: "volatility_1m", label: "1M Volatility" },
    { value: "drawdown_6m", label: "6M Drawdown" },
    { value: "rsi_14", label: "RSI 14" },
    { value: "ticker", label: "Ticker" },
];

const plotColors = {
    accent: "#ffb000",
    accentSoft: "rgba(255, 176, 0, 0.12)",
    cyan: "#3dc2ff",
    green: "#29d17d",
    red: "#ff5c66",
    text: "#d8dfeb",
    textSoft: "#8f9db3",
    grid: "rgba(111, 132, 160, 0.18)",
};

const BACKTEST_TIMEOUT_MS = 45000;
const STATUS_REFRESH_INTERVAL_MS = 30000;

const state = {
    universe: [],
    lastResult: null,
    universeRetryTimer: null,
    universeAutoRetryCount: 0,
    status: null,
    statusTimer: null,
};

const percentFormatter = new Intl.NumberFormat("en-IN", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
});
const numberFormatter = new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
});
const integerFormatter = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const currencyFormatter = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
});

document.addEventListener("DOMContentLoaded", async () => {
    hydrateSortOptions();
    hydrateGlossary();
    initializeDates();
    bindEvents();
    await loadUniverse();
});

function hydrateSortOptions() {
    const select = document.getElementById("sortBy");
    select.innerHTML = sortOptions
        .map((option) => `<option value="${option.value}">${option.label}</option>`)
        .join("");
}

function hydrateGlossary() {
    const container = document.getElementById("glossaryCards");
    container.innerHTML = glossaryItems
        .map(
            (item) => `
                <article class="note-card">
                    <span class="note-label">${item.term}</span>
                    <p>${item.plain}</p>
                </article>
            `
        )
        .join("");
}

function initializeDates() {
    const today = new Date();
    const startDate = new Date(today);
    startDate.setFullYear(startDate.getFullYear() - 1);
    document.getElementById("endDate").value = toIsoDate(today);
    document.getElementById("startDate").value = toIsoDate(startDate);
}

function bindEvents() {
    document.getElementById("sortBy").addEventListener("change", () => loadUniverse());
    document.getElementById("searchInput").addEventListener("input", () => renderUniverseTable());
    document.getElementById("presetTop5").addEventListener("click", () => presetSelection("top5"));
    document.getElementById("presetMomentum").addEventListener("click", () => presetSelection("momentum"));
    document.getElementById("presetLiquid").addEventListener("click", () => presetSelection("liquid"));
    document.getElementById("clearSelection").addEventListener("click", clearSelection);
    document.getElementById("runBacktestButton").addEventListener("click", runBacktest);
    document.getElementById("chartScale").addEventListener("change", () => {
        if (state.lastResult) {
            renderPerformanceChart(state.lastResult.performance);
        }
    });
}

async function loadUniverse() {
    const preservedSelection = getSelectedTickers();
    setStatus({
        tone: "loading",
        badge: "Syncing",
        title: "Fetching market data",
        detail: "Refreshing the current market snapshot for the active stock list.",
        activity: true,
        kind: "market-loading",
    });
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 12000);
    try {
        const sortBy = document.getElementById("sortBy").value;
        const response = await fetch(`/api/universe?sort_by=${encodeURIComponent(sortBy)}`, {
            signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Unable to load the market snapshot.");
        }

        state.universe = payload.stocks || [];
        syncDateInputsToSnapshot(payload.as_of);
        updateMarketTape(payload);
        const runtimeState = payload.is_fallback
            ? "Latest snapshot active"
            : payload.is_stale
              ? "Market data cached"
              : "Market data current";
        document.getElementById("runtimeStatus").textContent = `${runtimeState} · ${payload.as_of}`;
        document.getElementById("universeAsOf").textContent = `Data as of ${payload.as_of}`;
        renderUniverseTable(preservedSelection);
        if (payload.warning) {
            toast("Using the latest available market snapshot while fresh market data continues to refresh.", "warning");
        }
        setStatus({
            tone: payload.is_fallback || payload.is_stale ? "info" : "success",
            badge: payload.is_fallback || payload.is_stale ? "Updated" : "Live",
            title: "Market data updated",
            detail: payload.is_fallback || payload.is_stale
                ? "The latest available market snapshot is ready for basket selection."
                : "The current market snapshot is ready for basket selection.",
            freshnessAt: new Date(),
            activity: false,
            kind: "market-ready",
        });
        if (state.universeRetryTimer) {
            window.clearTimeout(state.universeRetryTimer);
            state.universeRetryTimer = null;
        }
        if (!payload.is_fallback) {
            state.universeAutoRetryCount = 0;
        }
        if (payload.is_fallback && state.universeAutoRetryCount < 1) {
            state.universeAutoRetryCount += 1;
            state.universeRetryTimer = window.setTimeout(() => {
                state.universeRetryTimer = null;
                loadUniverse();
            }, 13000);
        }
    } catch (error) {
        const message = error.name === "AbortError"
            ? "Market data refresh is taking longer than expected. Try again in a moment."
            : "Market data could not be refreshed. Try again to request a new snapshot.";
        toast(message, "danger");
        setStatus({
            tone: "error",
            badge: "Issue",
            title: "Market data needs attention",
            detail: message,
            activity: false,
            kind: "error",
        });
    } finally {
        window.clearTimeout(timeoutId);
    }
}

function updateMarketTape(payload) {
    const tape = payload.tape || {};
    document.getElementById("marketTape").innerHTML = `
        <span class="tape-item">As Of ${payload.as_of || "N/A"}</span>
        <span class="tape-item">Top 3M Momentum: ${tape.top_momentum || "N/A"}</span>
        <span class="tape-item">Avg 3M Momentum: ${formatPercent(tape.avg_3m_momentum)}</span>
        <span class="tape-item">Avg 1M Vol: ${formatPercent(tape.avg_1m_volatility)}</span>
    `;
}

function renderUniverseTable(preservedSelection = null) {
    const query = document.getElementById("searchInput").value.trim().toUpperCase();
    const tbody = document.getElementById("universeTableBody");
    const selected = preservedSelection || getSelectedTickers();
    const filtered = state.universe.filter((row) => {
        return !query || row.display_ticker.includes(query) || row.ticker.includes(query);
    });
    const defaultSelection = new Set(
        filtered
            .filter((row) => row.selectable !== false)
            .slice(0, 5)
            .map((row) => row.ticker)
    );

    tbody.innerHTML = filtered
        .map((row, index) => {
            const isSelectable = row.selectable !== false;
            const checked = isSelectable && (selected.size === 0 ? defaultSelection.has(row.ticker) : selected.has(row.ticker));
            return `
                <tr>
                    <td><input class="stock-checkbox" type="checkbox" value="${row.ticker}" ${checked ? "checked" : ""} ${isSelectable ? "" : "disabled"}></td>
                    <td><span class="ticker-pill">${row.display_ticker}</span></td>
                    <td>${formatCurrency(row.last_price)}</td>
                    <td class="${metricClass(row.momentum_1m)}">${formatPercent(row.momentum_1m)}</td>
                    <td class="${metricClass(row.momentum_3m)}">${formatPercent(row.momentum_3m)}</td>
                    <td>${formatPercent(row.volatility_1m)}</td>
                    <td class="${metricClass(row.drawdown_6m)}">${formatPercent(row.drawdown_6m)}</td>
                    <td>${formatTurnover(row.avg_turnover)}</td>
                </tr>
            `;
        })
        .join("");

    document.querySelectorAll(".stock-checkbox").forEach((checkbox) => {
        checkbox.addEventListener("change", updateSelectedCount);
    });
    updateSelectedCount();
}

function presetSelection(mode) {
    const selectableUniverse = state.universe.filter((row) => row.selectable !== false);
    const targetTickers = (() => {
        if (mode === "top5") {
            return selectableUniverse.slice(0, 5).map((row) => row.ticker);
        }
        if (mode === "momentum") {
            return [...selectableUniverse]
                .sort((left, right) => (right.momentum_3m || -Infinity) - (left.momentum_3m || -Infinity))
                .slice(0, 8)
                .map((row) => row.ticker);
        }
        return [...selectableUniverse]
            .sort((left, right) => (right.avg_turnover || -Infinity) - (left.avg_turnover || -Infinity))
            .slice(0, 8)
            .map((row) => row.ticker);
    })();

    document.querySelectorAll(".stock-checkbox").forEach((checkbox) => {
        checkbox.checked = targetTickers.includes(checkbox.value);
    });
    updateSelectedCount();
}

function clearSelection() {
    document.querySelectorAll(".stock-checkbox").forEach((checkbox) => {
        checkbox.checked = false;
    });
    updateSelectedCount();
}

function getSelectedTickers() {
    return new Set(
        Array.from(document.querySelectorAll(".stock-checkbox:checked")).map((checkbox) => checkbox.value)
    );
}

function updateSelectedCount() {
    document.getElementById("selectedCount").textContent = getSelectedTickers().size;
}

async function runBacktest() {
    const selected = Array.from(getSelectedTickers());
    if (selected.length < 3) {
        const message = "Select at least 3 stocks before running the backtest.";
        toast(message, "warning");
        setStatus({
            tone: "error",
            badge: "Issue",
            title: "Backtest needs attention",
            detail: message,
            activity: false,
            kind: "error",
        });
        return;
    }

    const button = document.getElementById("runBacktestButton");
    button.disabled = true;
    button.textContent = "Running...";
    setStatus({
        tone: "running",
        badge: "Active",
        title: "Running backtest",
        detail: "Calculating factors, refreshing history, and constructing the portfolio.",
        activity: true,
        kind: "backtest-running",
    });
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), BACKTEST_TIMEOUT_MS);

    const payload = {
        start_date: document.getElementById("startDate").value,
        end_date: document.getElementById("endDate").value,
        selected_stocks: selected,
        rebalance_frequency: Number(document.getElementById("rebalanceFrequency").value),
        max_stocks: Number(document.getElementById("maxHoldings").value),
        transaction_cost_bps: Number(document.getElementById("transactionCosts").value),
    };

    try {
        const response = await fetch("/api/backtest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Unable to complete the backtest.");
        }

        state.lastResult = result;
        document.getElementById("resultsSection").classList.remove("hidden");
        renderSummaries(result.summary);
        renderMetrics(result.metrics);
        renderPerformanceChart(result.performance);
        renderDrawdownChart(result.performance);
        renderAllocationChart(result.latest_allocation);
        renderFeatureChart(result.feature_importances);
        renderSignalTable(result.latest_signal_table);
        renderTradeTable(result.trade_log);
        renderRebalanceTable(result.rebalance_summary);
        setStatus({
            tone: "success",
            badge: "Complete",
            title: "Backtest successful",
            detail: "Results are ready for review.",
            freshnessAt: new Date(),
            runtimeMs: result.meta.runtime_ms || 0,
            activity: false,
            kind: "backtest-success",
        });
        toast("Backtest completed successfully.", "success");
    } catch (error) {
        const rawMessage = error.name === "AbortError"
            ? "Backtest took too long to respond. The first run can take longer while the local market-data cache warms up. Try the core 10 basket again once, or shorten the date range."
            : error.message;
        const message = toProductBacktestMessage(rawMessage);
        toast(message, "danger");
        setStatus({
            tone: "error",
            badge: "Issue",
            title: "Backtest needs attention",
            detail: message,
            activity: false,
            kind: "error",
        });
    } finally {
        window.clearTimeout(timeoutId);
        button.disabled = false;
        button.textContent = "Run Backtest";
    }
}

function syncDateInputsToSnapshot(asOf) {
    if (!asOf) {
        return;
    }

    const endInput = document.getElementById("endDate");
    if (!endInput.value || endInput.value > asOf) {
        endInput.value = asOf;
    }
}

function renderSummaries(summary) {
    document.getElementById("overviewSummary").textContent = summary.overview || "";
    document.getElementById("diagnosticsSummary").textContent = summary.diagnostics || "";
    document.getElementById("assessmentSummary").textContent = summary.assessment || "";
}

function renderMetrics(metrics) {
    setMetric("metricTotalReturn", metrics.total_return, true);
    setMetric("metricBenchmarkReturn", metrics.benchmark_total_return, true);
    setMetric("metricSharpe", metrics.sharpe_ratio, false);
    setMetric("metricSortino", metrics.sortino_ratio, false);
    setMetric("metricCalmar", metrics.calmar_ratio, false);
    setMetric("metricAlpha", metrics.alpha, false);
    setMetric("metricBeta", metrics.beta, false);
    setMetric("metricDrawdown", metrics.max_drawdown, true);
    setMetric("metricInfoRatio", metrics.information_ratio, false);
    setMetric("metricHitRate", metrics.trade_hit_rate, true);
}

function setMetric(id, value, isPercent) {
    const element = document.getElementById(id);
    element.textContent = isPercent ? formatPercent(value) : formatNumber(value);
    element.classList.remove("metric-positive", "metric-negative");
    if (typeof value === "number" && value > 0) {
        element.classList.add("metric-positive");
    } else if (typeof value === "number" && value < 0) {
        element.classList.add("metric-negative");
    }
}

function renderPerformanceChart(performance) {
    Plotly.react(
        "performanceChart",
        [
            {
                x: performance.dates,
                y: performance.portfolio_values,
                type: "scatter",
                mode: "lines",
                name: "Portfolio",
                line: { color: plotColors.accent, width: 2.8 },
                fill: "tozeroy",
                fillcolor: plotColors.accentSoft,
            },
            {
                x: performance.dates,
                y: performance.benchmark_values,
                type: "scatter",
                mode: "lines",
                name: "Benchmark",
                line: { color: plotColors.cyan, width: 2, dash: "dot" },
            },
        ],
        buildLayout({
            yaxis: {
                title: "Cumulative NAV",
                type: document.getElementById("chartScale").value,
                gridcolor: plotColors.grid,
                zeroline: false,
            },
            xaxis: { type: "date", gridcolor: plotColors.grid, zeroline: false },
        }),
        { responsive: true, displayModeBar: false }
    );
}

function renderDrawdownChart(performance) {
    Plotly.react(
        "drawdownChart",
        [
            {
                x: performance.dates,
                y: performance.drawdown_values,
                type: "scatter",
                mode: "lines",
                name: "Drawdown",
                line: { color: plotColors.red, width: 2 },
                fill: "tozeroy",
                fillcolor: "rgba(255, 92, 102, 0.14)",
            },
        ],
        buildLayout({
            yaxis: {
                title: "Drawdown",
                tickformat: ".0%",
                gridcolor: plotColors.grid,
                zeroline: false,
            },
            xaxis: { type: "date", gridcolor: plotColors.grid, zeroline: false },
            margin: { l: 56, r: 18, t: 14, b: 42 },
        }),
        { responsive: true, displayModeBar: false }
    );
}

function renderAllocationChart(allocation) {
    if (!allocation || allocation.length === 0) {
        Plotly.react(
            "allocationChart",
            [],
            buildLayout({
                annotations: [{ text: "No active holdings", showarrow: false, font: { color: plotColors.textSoft, size: 16 } }],
            }),
            { responsive: true, displayModeBar: false }
        );
        return;
    }

    Plotly.react(
        "allocationChart",
        [
            {
                type: "pie",
                labels: allocation.map((row) => row.display_ticker),
                values: allocation.map((row) => row.weight),
                hole: 0.56,
                textinfo: "label+percent",
                marker: {
                    colors: ["#ffb000", "#3dc2ff", "#29d17d", "#ff8a00", "#5a7fff", "#ff5c66", "#8dc63f", "#d98fff"],
                },
            },
        ],
        buildLayout({ showlegend: false, margin: { l: 10, r: 10, t: 10, b: 10 } }),
        { responsive: true, displayModeBar: false }
    );
}

function renderFeatureChart(featureImportances) {
    const rows = [...(featureImportances || [])].sort(
        (left, right) => Math.abs(right.coefficient) - Math.abs(left.coefficient)
    );

    Plotly.react(
        "featureChart",
        [
            {
                type: "bar",
                orientation: "h",
                x: rows.map((row) => row.coefficient),
                y: rows.map((row) => row.feature),
                marker: {
                    color: rows.map((row) => (row.coefficient >= 0 ? plotColors.green : plotColors.red)),
                },
            },
        ],
        buildLayout({
            xaxis: { gridcolor: plotColors.grid, zeroline: false },
            yaxis: { automargin: true },
            margin: { l: 110, r: 18, t: 14, b: 42 },
        }),
        { responsive: true, displayModeBar: false }
    );
}

function renderSignalTable(rows) {
    document.getElementById("signalTableBody").innerHTML = (rows || [])
        .map(
            (row) => `
                <tr>
                    <td>${row.ticker}</td>
                    <td><span class="tag-pill ${row.selected ? "is-selected" : ""}">${row.selected ? "Portfolio" : "Watch"}</span></td>
                    <td class="${metricClass(row.signal_score)}">${formatNumber(row.signal_score)}</td>
                    <td class="${metricClass(row.expected_return)}">${formatPercent(row.expected_return)}</td>
                    <td class="${metricClass(row.momentum_3m)}">${formatPercent(row.momentum_3m)}</td>
                    <td>${formatPercent(row.volatility_1m)}</td>
                    <td>${formatNumber(row.rsi_14)}</td>
                </tr>
            `
        )
        .join("");
}

function renderTradeTable(rows) {
    document.getElementById("tradeTableBody").innerHTML = (rows || [])
        .slice()
        .reverse()
        .map(
            (row) => `
                <tr>
                    <td>${row.ticker}</td>
                    <td>${row.entry_date}</td>
                    <td>${row.exit_date}</td>
                    <td>${formatPercent(row.weight)}</td>
                    <td class="${metricClass(row.expected_return)}">${formatPercent(row.expected_return)}</td>
                    <td class="${metricClass(row.realized_return)}">${formatPercent(row.realized_return)}</td>
                </tr>
            `
        )
        .join("");
}

function renderRebalanceTable(rows) {
    document.getElementById("rebalanceTableBody").innerHTML = (rows || [])
        .map(
            (row) => `
                <tr>
                    <td>${row.date}</td>
                    <td>${integerFormatter.format(row.eligible_count || 0)}</td>
                    <td>${integerFormatter.format(row.selected_count || 0)}</td>
                    <td>${formatPercent(row.turnover)}</td>
                    <td class="${metricClass(row.avg_signal)}">${formatNumber(row.avg_signal)}</td>
                </tr>
            `
        )
        .join("");
}

function buildLayout(extra) {
    return {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {
            family: "IBM Plex Sans, sans-serif",
            color: plotColors.textSoft,
        },
        legend: {
            orientation: "h",
            y: -0.18,
            font: { color: plotColors.textSoft },
        },
        margin: { l: 56, r: 18, t: 14, b: 42 },
        ...extra,
    };
}

function setStatus(status) {
    state.status = status;
    renderStatus();
    if (state.statusTimer) {
        window.clearInterval(state.statusTimer);
        state.statusTimer = null;
    }
    if (status?.freshnessAt) {
        state.statusTimer = window.setInterval(renderStatus, STATUS_REFRESH_INTERVAL_MS);
    }
}

function renderStatus() {
    const statusBox = document.getElementById("statusBox");
    const status = state.status || {
        tone: "info",
        badge: "Ready",
        title: "Ready",
        detail: "Select a basket and launch a research run.",
        activity: false,
        kind: "idle",
    };
    statusBox.dataset.tone = status.tone || "info";
    statusBox.dataset.activity = status.activity ? "active" : "idle";

    const header = document.createElement("div");
    header.className = "status-head";

    const label = document.createElement("span");
    label.className = "status-label";
    label.textContent = "System Status";

    const badge = document.createElement("span");
    badge.className = "status-badge";
    badge.textContent = status.badge || "Active";

    header.append(label, badge);

    const main = document.createElement("div");
    main.className = "status-main";

    const indicator = document.createElement("span");
    indicator.className = "status-indicator";
    indicator.setAttribute("aria-hidden", "true");

    const headline = document.createElement("strong");
    headline.textContent = status.title;

    main.append(indicator, headline);

    const body = document.createElement("p");
    body.className = "status-note";
    body.textContent = buildStatusDescription(status);

    statusBox.replaceChildren(header, main, body);
    statusBox.classList.remove("status-transition");
    void statusBox.offsetWidth;
    statusBox.classList.add("status-transition");
}

function buildStatusDescription(status) {
    if (status.kind === "market-ready" && status.freshnessAt) {
        return `Updated data ${formatRelativeTime(status.freshnessAt)}. ${status.detail}`;
    }
    if (status.kind === "backtest-success" && status.freshnessAt) {
        return `Completed ${formatRelativeTime(status.freshnessAt)} in ${formatDuration(status.runtimeMs || 0)}. ${status.detail}`;
    }
    return status.detail || "";
}

function formatRelativeTime(timestamp) {
    const elapsedMs = Math.max(0, Date.now() - timestamp.getTime());
    const minutes = Math.floor(elapsedMs / 60000);
    const hours = Math.floor(minutes / 60);

    if (minutes < 1) {
        return "just now";
    }
    if (minutes === 1) {
        return "1 minute ago";
    }
    if (minutes < 60) {
        return `${minutes} minutes ago`;
    }
    if (hours === 1) {
        return "1 hour ago";
    }
    if (hours < 24) {
        return `${hours} hours ago`;
    }

    const days = Math.floor(hours / 24);
    return days === 1 ? "1 day ago" : `${days} days ago`;
}

function formatDuration(runtimeMs) {
    if (runtimeMs >= 10000) {
        return `${Math.round(runtimeMs / 1000)} s`;
    }
    if (runtimeMs >= 1000) {
        return `${(runtimeMs / 1000).toFixed(1)} s`;
    }
    return `${runtimeMs} ms`;
}

function toProductBacktestMessage(message) {
    if (!message) {
        return "Backtest could not be completed. Review the inputs and try again.";
    }
    if (message.includes("Select at least 3 stocks")) {
        return message;
    }
    if (message.includes("rate-limited") || message.includes("temporarily busy")) {
        return "Market data retrieval is temporarily busy. Try again shortly or reduce the basket size.";
    }
    if (message.includes("took too long") || message.includes("timed out")) {
        return "Backtest is taking longer than expected. Try a smaller basket or a shorter date range.";
    }
    if (message.includes("unavailable")) {
        return "Required market history is not available for this run. Adjust the basket or try again shortly.";
    }
    return message;
}

function toast(message, tone = "success") {
    const region = document.getElementById("toastRegion");
    const toastElement = document.createElement("div");
    toastElement.className = `toast toast-${tone}`;
    toastElement.textContent = message;
    region.appendChild(toastElement);
    window.setTimeout(() => toastElement.remove(), 4200);
}

function metricClass(value) {
    if (typeof value !== "number") {
        return "";
    }
    if (value > 0) {
        return "metric-positive";
    }
    if (value < 0) {
        return "metric-negative";
    }
    return "";
}

function formatPercent(value) {
    if (typeof value !== "number") {
        return "N/A";
    }
    return percentFormatter.format(value);
}

function formatNumber(value) {
    if (typeof value !== "number") {
        return "N/A";
    }
    return numberFormatter.format(value);
}

function formatCurrency(value) {
    if (typeof value !== "number") {
        return "N/A";
    }
    return currencyFormatter.format(value);
}

function formatTurnover(value) {
    if (typeof value !== "number") {
        return "N/A";
    }
    return `${numberFormatter.format(value / 10000000)} Cr`;
}

function toIsoDate(dateValue) {
    const year = dateValue.getFullYear();
    const month = `${dateValue.getMonth() + 1}`.padStart(2, "0");
    const day = `${dateValue.getDate()}`.padStart(2, "0");
    return `${year}-${month}-${day}`;
}
