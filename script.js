// ============================================================
// GLOBAL VARIABLES
// ============================================================

let pcaData = null;
let pcaLoadings = null;
let rawMetrics = null;
let varianceData = null;
let clusterData = null;
let aiNarrative = null;
let isMethodologyActive = false;
let lastActiveView = 'pca';

// ============================================================
// DATA LOADING
// ============================================================

async function loadPCAData() {
    try {
        console.log('Loading data...');
        
        const [scoresRes, loadingsRes, metricsRes, varianceRes, clusterRes, narrativeRes] = await Promise.all([
            fetch('data/pca_scores_by_surface.json'),
            fetch('data/pca_loadings_by_surface.json'),
            fetch('data/tennis_metrics_by_player.json'),
            fetch('data/pca_variance_by_surface.json'),
            fetch('data/pca_clusters.json'),
            fetch('data/pca_crew_ai_narrative.json')
        ]);
        
        pcaData = await scoresRes.json();
        pcaLoadings = await loadingsRes.json();
        rawMetrics = await metricsRes.json();
        varianceData = await varianceRes.json();
        clusterData = await clusterRes.json();
        aiNarrative = await narrativeRes.json();

        console.log('All data loaded successfully');

        renderPlot('clay');
        
        setTimeout(() => {
            updateAIInsights('Clay', 'pca');
        }, 300);
        
    } catch (error) {
        console.error('Error loading PCA data:', error);
    }
}

// ============================================================
// HELPERS
// ============================================================

function formatPlayerName(fullName) {
    const parts = fullName.trim().split(' ');
    if (parts.length === 1) {
        return fullName;
    }
    const firstNameInitial = parts[0].charAt(0) + '.';
    const lastName = parts[parts.length - 1];
    return `${firstNameInitial} ${lastName}`;
}

function formatVariableName(variable) {
    const nameMap = {
        'winner_error_ratio': 'Winner/UE Ratio',
        'drop_shot_effectiveness': 'Drop Shot Effectiveness',
        'break_point_conversion_pct': 'Break Point Conversion %',
        'net_points_won_pct': 'Net Points Won %',
        'first_serve_points_won': '1st Serve Points Won %',
        'first_serve_pts_won': '1st Serve Points Won %',
        'second_serve_points_won_pct': '2nd Serve Points Won %',
        'ace_pct': 'Ace %',
        'double_fault_pct': 'Double Fault %',
        'break_points_saved_pct': 'Break Points Saved %',
        'unreturned_serve_rate': 'Unreturned Serve Rate',
        'first_serve_pct': '1st Serve %'
    };
    
    if (nameMap[variable]) {
        return nameMap[variable];
    }
    
    return variable.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function getSurfaceColor(surface = null) {
    const activeSurface = surface || document.querySelector('.surface-btn.active')?.dataset.surface || 'clay';
    
    const colorMap = {
        'clay': { hex: '#D97A52', rgb: '217, 122, 82', name: 'Clay Orange' },
        'grass': { hex: '#5A9E4D', rgb: '90, 158, 77', name: 'Grass Green' },
        'hard': { hex: '#4A9FC9', rgb: '74, 159, 201', name: 'Hard Blue' }
    };
    
    return colorMap[activeSurface] || colorMap['clay'];
}

function getPCInterpretation(surfaceName, component) {
    if (!pcaLoadings) return '';
    
    const loadingsForSurface = pcaLoadings.filter(d => d.surface === surfaceName);
    const pcKey = component === 1 ? 'PC1' : 'PC2';
    
    const variablesWithRawLoadings = loadingsForSurface.map(item => ({
        name: item.variable,
        loading: parseFloat(item[pcKey]),
        absLoading: Math.abs(parseFloat(item[pcKey]))
    }));
    
    variablesWithRawLoadings.sort((a, b) => b.absLoading - a.absLoading);
    
    if (variablesWithRawLoadings.length === 0) return '';
    
    const first = variablesWithRawLoadings[0];
    const second = variablesWithRawLoadings[1];
    
    const firstAbsLoading = first.absLoading;
    const secondAbsLoading = second ? second.absLoading : 0;
    
    if (firstAbsLoading > 0.75) {
        const varName = formatVariableName(first.name);
        if (firstAbsLoading > 0.85) {
            return `Driven mostly by ${varName}`;
        } else {
            return `Driven heavily by ${varName}`;
        }
    }
    
    if (secondAbsLoading > 0 && firstAbsLoading > secondAbsLoading * 2) {
        const varName = formatVariableName(first.name);
        return `Driven mostly by ${varName}`;
    }
    
    if (second && secondAbsLoading > 0) {
        const var1 = formatVariableName(first.name);
        const var2 = formatVariableName(second.name);
        return `Driven by ${var1} & ${var2}`;
    }
    
    const varName = formatVariableName(first.name);
    return `Driven by ${varName}`;
}

// ============================================================
// LABEL FILTERING
// ============================================================

function getLabelsToShow(allPoints, threshold = 0.8, microThreshold = 0.15) {
    const used = new Set();
    const processed = new Set();
    
    for (let i = 0; i < allPoints.length; i++) {
        if (processed.has(allPoints[i].player)) continue;
        
        const cluster = [allPoints[i]];
        for (let j = i + 1; j < allPoints.length; j++) {
            const dx = parseFloat(allPoints[i].PC1) - parseFloat(allPoints[j].PC1);
            const dy = parseFloat(allPoints[i].PC2) - parseFloat(allPoints[j].PC2);
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < threshold) {
                cluster.push(allPoints[j]);
            }
        }
        
        cluster.forEach(p => processed.add(p.player));
        
        if (cluster.length > 3) {
            cluster.sort((a, b) => parseFloat(b.PC2) - parseFloat(a.PC2));
            if (cluster.length >= 1) used.add(cluster[0].player);
            if (cluster.length >= 2) used.add(cluster[Math.floor(cluster.length / 2)].player);
            if (cluster.length >= 3) used.add(cluster[cluster.length - 1].player);
        } else {
            cluster.forEach(p => used.add(p.player));
        }
    }
    
    const microClusters = [];
    const processedMicro = new Set();
    
    for (let i = 0; i < allPoints.length; i++) {
        if (processedMicro.has(allPoints[i].player)) continue;
        
        const microCluster = [allPoints[i]];
        for (let j = i + 1; j < allPoints.length; j++) {
            const dx = parseFloat(allPoints[i].PC1) - parseFloat(allPoints[j].PC1);
            const dy = parseFloat(allPoints[i].PC2) - parseFloat(allPoints[j].PC2);
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < microThreshold) {
                microCluster.push(allPoints[j]);
            }
        }
        
        if (microCluster.length > 1) {
            microCluster.forEach(p => processedMicro.add(p.player));
            microClusters.push(microCluster);
        }
    }
    
    for (const microCluster of microClusters) {
        microCluster.sort((a, b) => parseFloat(b.PC2) - parseFloat(a.PC2));
        const keeper = microCluster[0].player;
        for (let i = 1; i < microCluster.length; i++) {
            used.delete(microCluster[i].player);
        }
        used.add(keeper);
    }
    
    const showMap = new Map();
    allPoints.forEach(p => {
        showMap.set(p.player, used.has(p.player));
    });
    return showMap;
}

// ============================================================
// CLUSTER ELLIPSES
// ============================================================

function addClusterCirclesToPlot(plotDivId, surface) {
    if (!clusterData) return;
    
    const surfaceKey = surface.charAt(0).toUpperCase() + surface.slice(1);
    
    const visData = clusterData.visualization?.[surfaceKey];
    if (!visData || !visData.clusters || visData.clusters.length === 0) return;
    
    const surfaceColor = getSurfaceColor();
    const shapes = [];
    
    for (let idx = 0; idx < visData.clusters.length; idx++) {
        const cluster = visData.clusters[idx];
        const ellipse = cluster.ellipse;
        
        if (!ellipse) continue;
        
        const centerX = ellipse.center_x;
        const centerY = ellipse.center_y;
        const rx = ellipse.radius_x;
        const ry = ellipse.radius_y;
        
        const kappa = 0.552284749831;
        const ox = rx * kappa;
        const oy = ry * kappa;
        
        const path = `M ${centerX - rx} ${centerY} ` +
                     `C ${centerX - rx} ${centerY + oy}, ${centerX - ox} ${centerY + ry}, ${centerX} ${centerY + ry} ` +
                     `C ${centerX + ox} ${centerY + ry}, ${centerX + rx} ${centerY + oy}, ${centerX + rx} ${centerY} ` +
                     `C ${centerX + rx} ${centerY - oy}, ${centerX + ox} ${centerY - ry}, ${centerX} ${centerY - ry} ` +
                     `C ${centerX - ox} ${centerY - ry}, ${centerX - rx} ${centerY - oy}, ${centerX - rx} ${centerY} Z`;
        
        shapes.push({
            type: 'path',
            path: path,
            xref: 'x',
            yref: 'y',
            line: {
                color: `rgba(${surfaceColor.rgb}, 0.85)`,
                width: 0.4
            },
            fillcolor: `rgba(${surfaceColor.rgb}, 0.04)`,
            opacity: 1,
            layer: 'below'
        });
    }
    
    if (shapes.length > 0) {
        Plotly.relayout(plotDivId, { shapes: shapes });
    }
}

// ============================================================
// PCA PLOT
// ============================================================

function renderPlot(surface) {
    if (!pcaData) return;
    
    const surfaceData = pcaData.filter(row => row.surface.toLowerCase() === surface);
    const surfaceName = surface.charAt(0).toUpperCase() + surface.slice(1);
    
    const pc1Values = surfaceData.map(d => parseFloat(d.PC1));
    const maxAbsolutePC1 = Math.max(...pc1Values.map(val => Math.abs(val)));
    const dynamicBuffer = maxAbsolutePC1 * 1.25;
    
    const showMap = getLabelsToShow(surfaceData, 0.8);
    
    function getPlayerText(player) {
        return showMap.get(player) ? formatPlayerName(player) : '';
    }
    
    const sinnerAlcaraz = surfaceData.filter(d => d.group === 'sinner_alcaraz');
    const big3 = surfaceData.filter(d => d.group === 'big_3');
    const others = surfaceData.filter(d => d.group === 'others');
    
    const traces = [];
    
    if (sinnerAlcaraz.length > 0) {
        traces.push({
            x: sinnerAlcaraz.map(d => parseFloat(d.PC1)),
            y: sinnerAlcaraz.map(d => parseFloat(d.PC2)),
            text: sinnerAlcaraz.map(d => getPlayerText(d.player)),
            hovertext: sinnerAlcaraz.map(d => formatPlayerName(d.player)),
            mode: 'markers+text',
            textposition: 'top center',
            textfont: { size: 9, color: '#FFFFFF' },
            name: 'Sinner / Alcaraz',
            marker: { color: '#FFCB06', size: 8, line: { color: '#1A1A2E', width: 1 } }
        });
    }
    
    if (big3.length > 0) {
        traces.push({
            x: big3.map(d => parseFloat(d.PC1)),
            y: big3.map(d => parseFloat(d.PC2)),
            text: big3.map(d => getPlayerText(d.player)),
            hovertext: big3.map(d => formatPlayerName(d.player)),
            mode: 'markers+text',
            textposition: 'top center',
            textfont: { size: 9, color: '#FFFFFF' },
            name: 'Big 3',
            marker: { color: '#FF5733', size: 8, line: { color: '#1A1A2E', width: 1 } }
        });
    }
    
    if (others.length > 0) {
        traces.push({
            x: others.map(d => parseFloat(d.PC1)),
            y: others.map(d => parseFloat(d.PC2)),
            text: others.map(d => getPlayerText(d.player)),
            hovertext: others.map(d => formatPlayerName(d.player)),
            mode: 'markers+text',
            textposition: 'top center',
            textfont: { size: 9, color: '#FFFFFF' },
            name: 'Others',
            marker: { color: '#E2DECE', size: 8, line: { color: '#1A1A2E', width: 1 } }
        });
    }
    
    let pc1MainText = 'PC1';
    let pc1SubText = '';
    let pc2MainText = 'PC2';
    let pc2SubText = '';
    
    if (varianceData && varianceData.length > 0) {
        const pc1Data = varianceData.find(d => d.surface === surfaceName && d.component === 1);
        const pc2Data = varianceData.find(d => d.surface === surfaceName && d.component === 2);
        
        if (pc1Data) {
            pc1MainText = `PC1 (${(pc1Data.explained_variance_ratio * 100).toFixed(1)}%)`;
            pc1SubText = getPCInterpretation(surfaceName, 1);
        }
        if (pc2Data) {
            pc2MainText = `PC2 (${(pc2Data.explained_variance_ratio * 100).toFixed(1)}%)`;
            pc2SubText = getPCInterpretation(surfaceName, 2);
        }
    }
    
    const xaxisTitle = pc1SubText ? `${pc1MainText}<br><span style="font-size: 10.5px; font-style: italic; opacity: 0.8;">${pc1SubText}</span>` : pc1MainText;
    const yaxisTitle = pc2SubText ? `${pc2MainText}<br><span style="font-size: 10.5px; font-style: italic; opacity: 0.8;">${pc2SubText}</span>` : pc2MainText;
    
    const layout = {
        xaxis: { 
            range: [-dynamicBuffer, dynamicBuffer],
            title: { 
                text: xaxisTitle, 
                font: { color: '#FFFFFF', size: 11 },
                standoff: 15
            },
            color: '#fff', 
            showline: false,
            zeroline: true,
            zerolinecolor: 'rgba(255, 255, 255, 0.4)',
            zerolinewidth: 0.8,
            showgrid: true,
            gridcolor: '#444',
            griddash: 'dot'
        },
        yaxis: { 
            title: { 
                text: yaxisTitle, 
                font: { color: '#FFFFFF', size: 11 },
                standoff: 15
            },
            color: '#fff', 
            showline: false,
            zeroline: true,
            zerolinecolor: 'rgba(255, 255, 255, 0.4)',
            zerolinewidth: 0.8,
            showgrid: true,
            gridcolor: '#444',
            griddash: 'dot'
        },
        plot_bgcolor: '#1A1A1A',
        paper_bgcolor: '#1A1A1A',
        font: { color: '#FFFFFF', size: 10 },
        hovermode: 'closest',
        showlegend: false,
        margin: { t: 15, r: 20, b: 50, l: 70 },
        autosize: true,
        shapes: []
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    const container = document.getElementById('pca-plot');
    if (container) {
        container.style.width = '100%';
        container.style.height = '100%';
    }
    
    Plotly.purge('pca-plot');
    Plotly.newPlot('pca-plot', traces, layout, config);
    
    setTimeout(() => {
        addClusterCirclesToPlot('pca-plot', surface);
    }, 100);
}

// ============================================================
// LOADINGS PLOT
// ============================================================

function renderLoadingsPlot(surface) {
    if (!pcaLoadings) return;
    
    const loadingsContainer = document.querySelector('.loadings-container');
    if (!loadingsContainer) return;
    
    const surfaceName = surface.charAt(0).toUpperCase() + surface.slice(1);
    const loadingsData = pcaLoadings.filter(d => d.surface === surfaceName);
    const surfaceColor = getSurfaceColor();
    const surfaceColorHex = surfaceColor.hex;

    if (loadingsData.length === 0) {
        loadingsContainer.innerHTML = '<p class="placeholder">No loadings data available for this surface</p>';
        return;
    }
    
    loadingsContainer.innerHTML = '';
    const plotDiv = document.createElement('div');
    plotDiv.id = 'loadings-plot';
    plotDiv.style.width = '100%';
    plotDiv.style.height = '100%';
    plotDiv.style.minHeight = '500px';
    loadingsContainer.appendChild(plotDiv);
    
    const maxAbsValue = Math.max(
        ...loadingsData.flatMap(d => [Math.abs(parseFloat(d.PC1)), Math.abs(parseFloat(d.PC2))])
    );
    const scaleFactor = maxAbsValue > 1 ? 0.9 / maxAbsValue : 0.9;
    
    const scaledPoints = loadingsData.map((item, idx) => {
        const pc1 = parseFloat(item.PC1);
        const pc2 = parseFloat(item.PC2);
        return {
            ...item,
            index: idx,
            scaledX: pc1 * scaleFactor,
            scaledY: pc2 * scaleFactor,
            pc1,
            pc2
        };
    });
    
    const proximityThreshold = 0.25;
    const labelPositionMap = new Map();
    
    const closePairs = [];
    for (let i = 0; i < scaledPoints.length; i++) {
        for (let j = i + 1; j < scaledPoints.length; j++) {
            const dx = scaledPoints[i].scaledX - scaledPoints[j].scaledX;
            const dy = scaledPoints[i].scaledY - scaledPoints[j].scaledY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < proximityThreshold) {
                closePairs.push({ i, j, distance });
            }
        }
    }
    
    for (const pair of closePairs) {
        const pointA = scaledPoints[pair.i];
        const pointB = scaledPoints[pair.j];
        
        const higherYPoint = pointA.scaledY > pointB.scaledY ? pointA : pointB;
        const lowerYPoint = pointA.scaledY > pointB.scaledY ? pointB : pointA;
        
        labelPositionMap.set(higherYPoint.index, {
            offset: 0.06,
            yanchor: 'bottom'
        });
        
        labelPositionMap.set(lowerYPoint.index, {
            offset: -0.06,
            yanchor: 'top'
        });
    }
    
    const arrowTraces = [];
    const annotations = [];
    
    for (let i = 0; i < scaledPoints.length; i++) {
        const point = scaledPoints[i];
        
        arrowTraces.push({
            x: [0, point.scaledX],
            y: [0, point.scaledY],
            mode: 'lines',
            line: { color: surfaceColorHex, width: 2 },
            showlegend: false,
            hoverinfo: 'text',
            hovertext: `${formatVariableName(point.variable)}<br>PC1: ${point.pc1.toFixed(3)}<br>PC2: ${point.pc2.toFixed(3)}`
        });
        
        let offsetY = 0;
        let yanchor = 'middle';
        
        if (labelPositionMap.has(i)) {
            offsetY = labelPositionMap.get(i).offset;
            yanchor = labelPositionMap.get(i).yanchor;
        } else {
            let offsetDistance = 0.06;
            
            if (point.scaledY > 0) {
                offsetY = offsetDistance;
                yanchor = 'bottom';
            } else if (point.scaledY < 0) {
                offsetY = -offsetDistance;
                yanchor = 'top';
            } else if (point.scaledX > 0) {
                offsetY = offsetDistance;
                yanchor = 'bottom';
            } else {
                offsetY = -offsetDistance;
                yanchor = 'top';
            }
        }
        
        annotations.push({
            x: point.scaledX,
            y: point.scaledY + offsetY,
            xref: 'x',
            yref: 'y',
            text: formatVariableName(point.variable),
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: surfaceColorHex,
            ax: 0,
            ay: 0,
            font: { size: 9, color: '#FFFFFF' },
            bgcolor: 'rgba(26,26,26,0.7)',
            yanchor: yanchor
        });
    }
    
    const theta = Array.from({ length: 100 }, (_, i) => (i / 99) * 2 * Math.PI);
    const circleX = theta.map(t => Math.cos(t));
    const circleY = theta.map(t => Math.sin(t));
    
    const traces = [
        {
            x: circleX,
            y: circleY,
            mode: 'lines',
            line: { color: 'rgba(255,255,255,0.4)', width: 1, dash: 'dot' },
            showlegend: false,
            hoverinfo: 'none'
        },
        ...arrowTraces,
        {
            x: scaledPoints.map(d => d.scaledX),
            y: scaledPoints.map(d => d.scaledY),
            mode: 'markers',
            marker: { color: surfaceColorHex, size: 6, line: { color: surfaceColorHex, width: 1 } },
            showlegend: false,
            hoverinfo: 'skip'
        }
    ];
    
    let pc1MainText = 'PC1';
    let pc1SubText = '';
    let pc2MainText = 'PC2';
    let pc2SubText = '';
    
    if (varianceData && varianceData.length > 0) {
        const pc1Data = varianceData.find(d => d.surface === surfaceName && d.component === 1);
        const pc2Data = varianceData.find(d => d.surface === surfaceName && d.component === 2);
        
        if (pc1Data) {
            pc1MainText = `PC1 (${(pc1Data.explained_variance_ratio * 100).toFixed(1)}%)`;
            pc1SubText = getPCInterpretation(surfaceName, 1);
        }
        if (pc2Data) {
            pc2MainText = `PC2 (${(pc2Data.explained_variance_ratio * 100).toFixed(1)}%)`;
            pc2SubText = getPCInterpretation(surfaceName, 2);
        }
    }
    
    const xaxisTitle = pc1SubText ? `${pc1MainText}<br><span style="font-size: 10.5px; font-style: italic; opacity: 0.8;">${pc1SubText}</span>` : pc1MainText;
    const yaxisTitle = pc2SubText ? `${pc2MainText}<br><span style="font-size: 10.5px; font-style: italic; opacity: 0.8;">${pc2SubText}</span>` : pc2MainText;
    
    const layout = {
        title: {
            text: `Variable Projections on PC1 and PC2 - ${surfaceName}`,
            font: { color: '#FFFFFF', size: 14, family: 'Inter' },
            x: 0.5
        },
        xaxis: {
            title: { 
                text: xaxisTitle, 
                font: { color: '#FFFFFF', size: 11 },
                standoff: 15
            },
            range: [-1.1, 1.1],
            zeroline: true,
            zerolinecolor: 'rgba(255,255,255,0.5)',
            zerolinewidth: 1,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.1)',
            color: '#fff',
            showline: false
        },
        yaxis: {
            title: { 
                text: yaxisTitle, 
                font: { color: '#FFFFFF', size: 11 },
                standoff: 15
            },
            range: [-1.1, 1.1],
            zeroline: true,
            zerolinecolor: 'rgba(255,255,255,0.5)',
            zerolinewidth: 1,
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.1)',
            scaleanchor: 'x',
            scaleratio: 1,
            color: '#fff',
            showline: false
        },
        plot_bgcolor: '#1A1A1A',
        paper_bgcolor: '#1A1A1A',
        font: { color: '#FFFFFF', size: 10 },
        annotations: annotations,
        margin: { t: 60, r: 30, b: 70, l: 90 },
        autosize: true,
        hovermode: 'closest',
        showlegend: false
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.purge('loadings-plot');
    Plotly.newPlot('loadings-plot', traces, layout, config);
}

// ============================================================
// METRICS TABLE
// ============================================================

function renderMetricsTable(surface) {
    if (!rawMetrics) return;
    
    const metricsContainer = document.querySelector('.metrics-container');
    if (!metricsContainer) return;
    
    const selectedSurface = surface || document.querySelector('.surface-btn.active').dataset.surface;
    const surfaceName = selectedSurface.charAt(0).toUpperCase() + selectedSurface.slice(1);
    
    const filteredMetrics = rawMetrics.filter(row => row.surface === surfaceName);
    const metricKeys = Object.keys(rawMetrics[0]).filter(k => !['surface', 'player'].includes(k));
    
    metricsContainer.innerHTML = '';
    
    const wrapper = document.createElement('div');
    wrapper.style.width = '100%';
    wrapper.style.height = '100%';
    
    const table = document.createElement('table');
    table.className = 'premium-table';
    table.style.minWidth = '800px';
    table.style.width = '100%';
    
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    const thPlayer = document.createElement('th');
    thPlayer.textContent = 'Player';
    headerRow.appendChild(thPlayer);
    
    metricKeys.forEach(key => {
        const th = document.createElement('th');
        const rawName = formatVariableName(key);
        
        const words = rawName.split(' ');
        let displayName = rawName;
        
        if (words.length >= 3) {
            const firstLine = words.slice(0, 2).join(' ');
            const secondLine = words.slice(2).join(' ');
            displayName = `${firstLine}<br>${secondLine}`;
        }
        
        th.innerHTML = displayName;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    const tbody = document.createElement('tbody');
    filteredMetrics.forEach(row => {
        const tr = document.createElement('tr');
        
        const tdPlayer = document.createElement('td');
        tdPlayer.textContent = row.player;
        tr.appendChild(tdPlayer);
        
        metricKeys.forEach(key => {
            const td = document.createElement('td');
            td.textContent = parseFloat(row[key]).toFixed(3);
            tr.appendChild(td);
        });
        
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    wrapper.appendChild(table);
    metricsContainer.appendChild(wrapper);
}

// ============================================================
// SURFACE SWITCHING
// ============================================================

function setActiveSurface(surface) {
    const leftPanel = document.getElementById('left-panel');
    const rightPanel = document.querySelector('.right-panel');
    
    leftPanel.classList.remove('clay-bg', 'grass-bg', 'hard-bg');
    leftPanel.classList.add(`${surface}-bg`);
    
    document.querySelectorAll('.surface-btn').forEach(btn => {
        if (btn.dataset.surface === surface) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    rightPanel.classList.remove('clay-active-indicator', 'grass-active-indicator', 'hard-active-indicator');
    rightPanel.classList.add(`${surface}-active-indicator`);
    
    const activeView = document.querySelector('.view-btn.active').dataset.view;
    if (activeView === 'loadings') {
        renderLoadingsPlot(surface);
    } else if (activeView === 'metrics') {
        renderMetricsTable(surface);
    }
    
    updateAIInsights(surface, activeView);
}

// ============================================================
// VIEW SWITCHING
// ============================================================

function setActiveView(view) {
    if (isMethodologyActive) return;
    
    const pcaView = document.getElementById('pca-view');
    const loadingsView = document.getElementById('loadings-view');
    const metricsView = document.getElementById('metrics-view');
    
    document.querySelectorAll('.view-btn').forEach(btn => {
        if (btn.dataset.view === view) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    pcaView.style.display = view === 'pca' ? 'flex' : 'none';
    loadingsView.style.display = view === 'loadings' ? 'flex' : 'none';
    metricsView.style.display = view === 'metrics' ? 'flex' : 'none';
    
    const currentSurface = document.querySelector('.surface-btn.active').dataset.surface;
    
    if (view === 'metrics') {
        renderMetricsTable(currentSurface);
    }
    if (view === 'loadings') {
        renderLoadingsPlot(currentSurface);
    }
    if (view === 'pca') {
        setTimeout(() => {
            renderPlot(currentSurface);
        }, 50);
    }
    
    updateAIInsights(currentSurface, view);
}

// ============================================================
// METHODOLOGY
// ============================================================

function showMethodologyView() {
    const mainContent = document.getElementById('main-content');
    const methodologyView = document.getElementById('methodology-view');
    const topBar = document.querySelector('.top-bar');
    const rightPanel = document.querySelector('.right-panel');
    
    const activeBtn = document.querySelector('.view-btn.active');
    if (activeBtn) {
        lastActiveView = activeBtn.dataset.view;
    }
    
    mainContent.style.display = 'none';
    topBar.style.display = 'none';
    rightPanel.style.padding = '0';
    methodologyView.style.display = 'flex';
    
    isMethodologyActive = true;
    
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    renderMethodologyContent();
}

function hideMethodologyView() {
    const mainContent = document.getElementById('main-content');
    const methodologyView = document.getElementById('methodology-view');
    const topBar = document.querySelector('.top-bar');
    const rightPanel = document.querySelector('.right-panel');
    
    methodologyView.style.display = 'none';
    mainContent.style.display = 'flex';
    topBar.style.display = 'flex';
    rightPanel.style.padding = '';
    
    isMethodologyActive = false;
    
    const currentSurface = document.querySelector('.surface-btn.active').dataset.surface;
    
    const viewBtn = document.querySelector(`.view-btn[data-view="${lastActiveView}"]`);
    if (viewBtn) {
        viewBtn.classList.add('active');
    }
    
    const pcaView = document.getElementById('pca-view');
    const loadingsView = document.getElementById('loadings-view');
    const metricsView = document.getElementById('metrics-view');
    
    pcaView.style.display = 'none';
    loadingsView.style.display = 'none';
    metricsView.style.display = 'none';
    
    if (lastActiveView === 'pca') {
        pcaView.style.display = 'flex';
        setTimeout(() => {
            renderPlot(currentSurface);
            setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
        }, 50);
    } else if (lastActiveView === 'loadings') {
        loadingsView.style.display = 'flex';
        setTimeout(() => {
            renderLoadingsPlot(currentSurface);
            setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
        }, 50);
    } else if (lastActiveView === 'metrics') {
        metricsView.style.display = 'flex';
        renderMetricsTable(currentSurface);
    }
}

function renderMethodologyContent() {
    const methodologyContainer = document.querySelector('.methodology-container');
    if (!methodologyContainer) return;
    
    methodologyContainer.classList.add('has-more-content');
    
    const activeSurface = document.querySelector('.surface-btn.active').dataset.surface;
    let surfaceColor = '#D97A52';
    
    if (activeSurface === 'grass') {
        surfaceColor = '#5A9E4D';
    } else if (activeSurface === 'hard') {
        surfaceColor = '#4A9FC9';
    }
    
    methodologyContainer.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <h2 style="margin: 0; color: ${surfaceColor};">Methodology</h2>
            <button class="methodology-back-btn" id="back-from-methodology" aria-label="Back to Analysis">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style="stroke: ${surfaceColor};">
                    <path d="M15 18L9 12L15 6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
        </div>
        
        <h3 style="color: ${surfaceColor};" class="methodology-h3">Data Overview</h3>
        <p class="methodology-text">To characterize elite tennis playing styles, this study is based on granular tracking records from the Tennis Match Charting Project (MCP) dataset, maintained by Jeff Sackmann. The shot‑level variables in the MCP data (such as net approach frequency, slice rates, and shot direction) are particularly relevant for defining playing pattern profiles. However, the charting data is sparse before 2014, introducing a temporal bias. In addition, grass‑court matches are naturally limited, while superstars like Novak Djokovic have far more tracked matches than their peers. Under standard pooling, these disparities would heavily distort statistical variance.</p>
        
        <h3 style="color: ${surfaceColor};" class="methodology-h3">Sampling Strategy</h3>
        <p class="methodology-text">The target player pool consists of players who achieved a year‑end Top 10 ATP ranking between 2010 and 2024. To ensure data quality and analytical and conceptual relevance, the match scope is restricted to 2010 onward at Grand Slams, Masters 1000s, and premier grass events from the Round of 32 onward. Given the structural distortions caused by uneven tracking volumes across eras and surfaces within the Sackmann dataset coupled with corrupt or incomplete data logs, strict inclusion criteria and sampling equalization were applied. Players from the Top 10 cohort with a minimum of five charted matches on each of the three traditional surfaces (hard, clay, grass) were included, reducing the final working pool to 21 players.</p>
        
        <h3 style="color: ${surfaceColor};" class="methodology-h3">PCA Variable Selection & Optimization</h3>
        <p class="methodology-text">From this balanced dataset, 14 continuous metrics are calculated. Each metric is expressed as a rate or percentage using its own appropriate denominator: total shots for slice rate and drop shot frequency, serve points for ace rate and unreturned serve rate, net points for net points won, and so on. For drop shots, the approach separately measures frequency (drop shots per total shot) and effectiveness (winners plus induced forced errors per drop shot). Other metrics capture either tactical frequency (such as slice rate and crosscourt ratio) or execution quality (such as ace rate, net points won, and winner/error ratio), each normalized to its relevant context. Advanced variables capture strategic nuance beyond simple counts: for example, Shannon entropy calculated over serve direction percentages (wide, body, T) measures placement variety rather than raw power or ace totals. The final set of metrics is as follows: Winner Error Ratio, Break Points Saved Pct, Ace Rate, First Serve Pct, First Serve Pts Won, Unreturned Serve Rate, Serve Placement Variety, Return Points Won Pct, Break Point Conversion Pct, Net Points Won Pct, Crosscourt Ratio, Slice Rate, Drop Shot Frequency, and Drop Shot Effectiveness. A four-step optimization process was applied independently to each surface:</p>

        <ul class="methodology-text" style="margin-left: 2rem; padding-left: 0;margin-top: 0.5rem; margin-bottom: 0.5rem">
            <li><strong>Step 1.</strong> Tennis-informed (8 vars): Expert-selected variables based on tennis knowledge.</li>
            <li><strong>Step 2.</strong> Assumption validator (4 vars): Variables selected by statistical rules (KMO, variance).</li>
            <li><strong>Step 3.</strong> Exhaustive search (p = 4 to p = 8): Tested all C(14,4) to C(14,8) = 12,441 combinations per surface.</li>
            <li><strong>Step 4.</strong> Optimized (4 vars): Best combination from exhaustive search.</li>
        </ul>
        
        <p class="methodology-text">Principal Component Analysis (PCA) variable selection was then optimized per surface. For Clay and Grass, variables were selected using statistical rules (variance thresholds), achieving n/p=5.25:1 and capturing 63-64% of variance. For Hard court, an exhaustive search identified a superior set of variables (n/p=5.25:1, variance=90.2%, KMO=0.624). This mixed approach balances statistical adequacy with explained variance.</p>

        <h3 style="color: ${surfaceColor};font-size: 0.8rem;font-weight: 600;margin-top: 1.25rem; margin-bottom: 0.5rem" class="table-title">Table 1: PCA Optimization Results by Surface</h3>
        
        <table class="methodology-table">
            <thead>
                <tr>
                    <th>Surface</th>
                    <th>Variables</th>
                    <th>PC1</th>
                    <th>PC2</th>
                    <th>Total</th>
                    <th>n/p</th>
                    <th>KMO</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="color: #FFFFFF; font-weight: 600;">Clay</td>
                    <td>Winner Error Ratio, Drop Shot Effectiveness, Break Point Conversion Pct, First Serve Pct</td>
                    <td>38.2%</td>
                    <td>25.1%</td>
                    <td>63.3%</td>
                    <td>5.25:1</td>
                    <td>0.577</td>
                </tr>
                <tr>
                    <td style="color: #FFFFFF; font-weight: 600;">Grass</td>
                    <td>Winner Error Ratio, Drop Shot Effectiveness, Break Point Conversion Pct, Break Points Saved Pct</td>
                    <td>40.1%</td>
                    <td>24.2%</td>
                    <td>64.3%</td>
                    <td>5.25:1</td>
                    <td>0.582</td>
                </tr>
                <tr>
                    <td style="color: #FFFFFF; font-weight: 600;">Hard</td>
                    <td>Winner Error Ratio, Break Points Saved Pct, First Serve Points Won, Unreturned Serve Rate</td>
                    <td>66.4%</td>
                    <td>23.8%</td>
                    <td>90.2%</td>
                    <td>5.25:1</td>
                    <td>0.624</td>
                </tr>
            </tbody>
        </table>

        <h3 style="color: ${surfaceColor};" class="methodology-h3">Clustering Analysis</h3>
        <p class="methodology-text">To identify natural groupings of playing styles, hierarchical agglomerative clustering with complete-linkage was applied independently to each surface. Complete-linkage defines cluster distance as the maximum distance between points in different clusters, producing compact, interpretable groupings with similar diameters, which is ideal for visualizing tactical profiles on PCA biplots. The optimal number of clusters was determined using silhouette scores across k = 2 to k = 6 solutions. For Grass courts, a domain-specific safeguard enforces a minimum of three clusters when sample size permits, preventing the silhouette criterion from artificially flattening tactical heterogeneity into a binary solution. A silhouette threshold of 0.22 is also applied to favor three-cluster solutions where statistically defensible. Single-player clusters are treated as outliers, while ellipses are fitted around multi-player clusters using point cloud bounds for clear spatial visualization.</p>

        <h3 style="color: ${surfaceColor};" class="methodology-h3">Single Agent vs Multi-Agent Debate</h3>
        <p class="methodology-text">Multi-agent systems introduce unnecessary complexity, reliability issues, and security risks (CVE-2026-2275, CVE-2026-2285, CVE-2026-2286, CVE-2026-2287). A single agent with a well-crafted prompt and strict formatting rules is more robust, cost-effective, and deterministic. PCA and clustering are purely statistical methods requiring no AI agent intervention.</p>

        <h3 style="color: ${surfaceColor};" class="methodology-h3">Methodological Safeguards</h3>
        <p class="methodology-text">A set of protocols was embedded into the instruction framework to ensure mathematical accuracy in AI-generated PCA narratives:</p>
        <ul class="methodology-text" style="margin-left: 2rem; padding-left: 0; margin-top: 0.5rem; margin-bottom: 0.5rem">
            <li><strong>Pre-Writing Protocol:</strong> Extract loadings, identify primary axis, directional sign, and secondary influences (>0.3) for each metric before writing.</li>
            <li><strong>Directional Sign Rule:</strong> Positive loadings push toward the positive end; negative loadings pull toward the negative end. Metrics are explicitly separated by sign when defining an axis.</li>
            <li><strong>Primary-Secondary Loading Rule:</strong> Metrics are described on their primary axis; secondary influences (>0.3) are acknowledged; loadings below 0.3 are ignored.</li>
            <li><strong>One-Sided Axis Rule:</strong> Axes with exclusively one-sided loadings are characterized as structural deficits, not opposing styles.</li>
            <li><strong>Coordinate-to-Metric Protocol:</strong> Player positions are interpreted using the sign and magnitude of loadings on each axis, preventing cross-assignment.</li>
            <li><strong>Surface Context Rule:</strong> High-touch metrics are interpreted as patient play on Clay and high-risk gambles on Grass.</li>
            <li><strong>Tone Guardrails:</strong> Human analyst voice, no self-reference, no jargon, natural visual geography.</li>
        </ul>

        <h3 style="color: ${surfaceColor};" class="methodology-h3">Limitations</h3>
        <p class="methodology-text">Due to the elite sample constraint (n = 21), statistical power is limited. This is most apparent on Grass, where the Step 4 solution showed low KMO (0.466) and was rejected in favor of the statistically robust Step 2 solution. Additional limitations include temporal bias from sparse pre‑2014 data and residual uneven tracking volumes across players. Furthermore, the findings represent a player's core style derived from a balanced sample, not an exhaustive career history. The inherent subjectivity of manual charting, particularly for ambiguous shot classifications, should also be acknowledged. Given these constraints, all results should be interpreted as exploratory patterns rather than causal inferences, and findings require replication on independent datasets.</p>

        <h3 style="color: ${surfaceColor};" class="methodology-h3">Interpretation</h3>
        <p class="methodology-text">The four optimized metrics per surface are reduced to two principal components that capture the most variance in playing styles. Points closer together on the PCA biplot indicate similar playing styles. The axis titles show which metrics drive each component. The Scores section translates these components into tactical tennis language, while the Loadings section details the specific metric contributions.</p>
    `;
    
    const backBtn = document.getElementById('back-from-methodology');
    if (backBtn) {
        backBtn.addEventListener('click', hideMethodologyView);
    }
    
    setTimeout(initMethodologyScrollMask, 100);
}

// ============================================================
// AI INSIGHTS
// ============================================================

function updateAIInsights(surface, view) {
    const insightsText = document.getElementById('ai-insights-text');
    
    if (!insightsText) return;
    
    if (!aiNarrative) {
        insightsText.textContent = 'Loading AI insights...';
        return;
    }
    
    if (!aiNarrative.surface) {
        insightsText.textContent = 'AI narrative data format error.';
        return;
    }
    
    const surfaceKey = surface.charAt(0).toUpperCase() + surface.slice(1);
    const surfaceNarrative = aiNarrative.surface[surfaceKey];
    
    if (!surfaceNarrative) {
        insightsText.textContent = `No AI insights available for ${surfaceKey}.`;
        return;
    }
    
    if (view === 'pca') {
        insightsText.innerHTML = surfaceNarrative.scores_narrative || 'No scores narrative available.';
    } else if (view === 'loadings') {
        insightsText.innerHTML = surfaceNarrative.loadings_narrative || 'No loadings narrative available.';
    } else {
        insightsText.textContent = 'AI Insights are only available in PCA Biplot or Loadings view.';
    }
    
    insightsText.scrollTop = 0;
    setTimeout(initScrollMask, 100);
}

// ============================================================
// SCROLL MASKS
// ============================================================

function initScrollMask() {
    const insightsText = document.getElementById('ai-insights-text');
    if (!insightsText) return;
    
    insightsText.removeEventListener('scroll', handleScrollMask);
    insightsText.addEventListener('scroll', handleScrollMask);
    setTimeout(handleScrollMask, 50);
}

function handleScrollMask() {
    const insightsText = document.getElementById('ai-insights-text');
    if (!insightsText) return;
    
    const scrollTop = insightsText.scrollTop;
    const scrollHeight = insightsText.scrollHeight;
    const clientHeight = insightsText.clientHeight;
    
    if (scrollTop + clientHeight >= scrollHeight - 5) {
        insightsText.classList.remove('has-more-content');
    } else {
        insightsText.classList.add('has-more-content');
    }
}

function initMethodologyScrollMask() {
    const container = document.querySelector('.methodology-full .methodology-container');
    if (!container) return;
    
    container.removeEventListener('scroll', handleMethodologyScroll);
    container.addEventListener('scroll', handleMethodologyScroll);
    setTimeout(handleMethodologyScroll, 50);
}

function handleMethodologyScroll() {
    const container = document.querySelector('.methodology-full .methodology-container');
    if (!container) return;
    
    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight;
    const clientHeight = container.clientHeight;
    
    if (scrollTop + clientHeight >= scrollHeight - 5) {
        container.classList.remove('has-more-content');
    } else {
        container.classList.add('has-more-content');
    }
}

// ============================================================
// WELCOME OVERLAY
// ============================================================

function initWelcome() {
    const overlay = document.getElementById('welcome-overlay');
    if (!overlay) return;
    
    overlay.classList.add('active');
    
    function dismiss() {
        overlay.classList.remove('active');
        overlay.style.display = 'none';
    }
    
    document.getElementById('welcome-get-started').addEventListener('click', dismiss);
    document.getElementById('welcome-close-btn').addEventListener('click', dismiss);
    
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) dismiss();
    });
    
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && overlay.classList.contains('active')) dismiss();
    });
}

// ============================================================
// EVENT LISTENERS
// ============================================================

document.querySelectorAll('.surface-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const surface = this.dataset.surface;
        setActiveSurface(surface);
        renderPlot(surface);
    });
});

document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        setActiveView(this.dataset.view);
    });
});

document.getElementById('method-btn').addEventListener('click', showMethodologyView);

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    initWelcome();
    setActiveSurface('clay');
    setActiveView('pca');
    loadPCAData();
});