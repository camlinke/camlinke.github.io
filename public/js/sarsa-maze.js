// Episodic SARSA(lambda) with replacing eligibility traces, learning to solve
// a randomly generated maze. Rendered on <canvas id="sarsa-maze">.
(function () {
  'use strict';

  var canvas = document.getElementById('sarsa-maze');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');
  var caption = document.getElementById('sarsa-maze-caption');

  // ---- Configuration ------------------------------------------------------
  var ROWS = 20, COLS = 50, N = ROWS * COLS;
  var ALPHA = 0.7;          // step size (deterministic maze, so a large step is fine)
  var GAMMA = 0.99;         // discount
  var LAMBDA = 0.95;        // trace decay
  var EPSILON = 0.02;       // epsilon-greedy exploration
  var GOAL_REWARD = 1;      // reward on reaching the goal; every other step is 0
  var TRACE_MIN = 1e-3;     // traces below this are dropped
  var LOOP_FRACTION = 0.06; // extra walls removed so the maze has loops
  var MAX_FRAME_MS = 250;   // cap on catch-up time after a backgrounded tab
  // Agent steps per second for each speed: regular, fast (10x), fastest (100x).
  var SPEEDS = [
    { label: 'regular', rate: 2 },
    { label: 'fast', rate: 20 },
    { label: 'fastest', rate: 200 }
  ];

  // Site palette (Poole theme / _sass/_base.scss)
  var COLOR = {
    bg: '#ffffff',
    wall: '#515151',
    agent: '#268bd2',     // site link blue
    goal: '#859900',      // solarized green, pairs with the link blue
    trace: '38,139,210',  // agent blue as rgb, for eligibility-trace overlay
    value: '133,153,0',   // goal green as rgb, for learned-value overlay
    explored: '#f9f9f9'   // site's light table-stripe grey
  };

  // Actions / directions: N, E, S, W. Wall bits use the same order.
  var NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3;
  var DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];
  var WALL = [1, 2, 4, 8], OPP = [SOUTH, WEST, NORTH, EAST];
  var ALL_WALLS = 15;

  var START = 0, DEFAULT_GOAL = N - 1;

  // ---- State --------------------------------------------------------------
  var walls, goal;                         // environment
  var Q, E, active, inActive, visited;     // learner (E is sparse via `active`)
  var s, a, episode, steps, lastSteps, bestSteps;
  var running = false, rafId = null, speed = 0, pending = 0, lastTime = null;

  // ---- Grid helpers -------------------------------------------------------
  function rowOf(i) { return (i / COLS) | 0; }
  function colOf(i) { return i % COLS; }
  function inBounds(r, c) { return r >= 0 && r < ROWS && c >= 0 && c < COLS; }

  // Neighbouring cell in direction d, or -1 if off the grid.
  function neighbour(i, d) {
    var r = rowOf(i) + DR[d], c = colOf(i) + DC[d];
    return inBounds(r, c) ? r * COLS + c : -1;
  }

  // Remove the wall between cell i and its neighbour j in direction d.
  function carve(i, j, d) {
    walls[i] &= ~WALL[d];
    walls[j] &= ~WALL[OPP[d]];
  }

  function randInt(n) { return (Math.random() * n) | 0; }

  // ---- Maze generation (randomized DFS + a few extra openings) -------------
  function makeMaze() {
    walls = new Uint8Array(N).fill(ALL_WALLS);
    var seen = new Uint8Array(N), stack = [START], d, j;
    seen[START] = 1;
    while (stack.length) {
      var cur = stack[stack.length - 1], opts = [];
      for (d = 0; d < 4; d++) {
        j = neighbour(cur, d);
        if (j >= 0 && !seen[j]) opts.push(d);
      }
      if (!opts.length) { stack.pop(); continue; }
      d = opts[randInt(opts.length)];
      j = neighbour(cur, d);
      carve(cur, j, d);
      seen[j] = 1;
      stack.push(j);
    }

    // Knock out extra interior walls so there's more than one route.
    var toOpen = Math.floor(N * LOOP_FRACTION);
    while (toOpen > 0) {
      var i = randInt(N);
      d = randInt(4);
      j = neighbour(i, d);
      if (j < 0 || !(walls[i] & WALL[d])) continue;
      carve(i, j, d);
      toOpen--;
    }
  }

  // ---- Environment --------------------------------------------------------
  function move(state, action) {
    if (walls[state] & WALL[action]) return state;
    return state + DR[action] * COLS + DC[action];
  }

  // ---- Agent --------------------------------------------------------------
  function greedy(state) {
    var base = state * 4, best = -Infinity, n = 0, choice = 0;
    for (var i = 0; i < 4; i++) {
      var q = Q[base + i];
      if (q > best) { best = q; choice = i; n = 1; }
      else if (q === best && Math.random() * ++n < 1) choice = i; // random tie-break
    }
    return choice;
  }

  function policy(state) {
    return Math.random() < EPSILON ? randInt(4) : greedy(state);
  }

  function maxQ(state) {
    var base = state * 4, v = 0;
    for (var i = 0; i < 4; i++) if (Q[base + i] > v) v = Q[base + i];
    return v;
  }

  function clearTraces() {
    for (var i = 0; i < active.length; i++) { E[active[i]] = 0; inActive[active[i]] = 0; }
    active.length = 0;
  }

  function startEpisode() {
    clearTraces();
    s = START;
    a = policy(s);
    steps = 0;
    visited[s] = 1;
  }

  function endEpisode() {
    lastSteps = steps;
    bestSteps = bestSteps === null ? steps : Math.min(bestSteps, steps);
    episode++;
    startEpisode();
  }

  function resetAgent() {
    Q = new Float32Array(N * 4);   // zero init: neutral, so the agent explores until it finds reward
    E = new Float32Array(N * 4);
    inActive = new Uint8Array(N * 4);
    visited = new Uint8Array(N);
    active = [];
    episode = 1; lastSteps = null; bestSteps = null;
    startEpisode();
  }

  // One SARSA(lambda) transition. Returns true when the episode ends.
  function step() {
    var s2 = move(s, a), sa = s * 4 + a;
    var terminal = s2 === goal;
    var a2 = terminal ? 0 : policy(s2);
    var target = terminal ? GOAL_REWARD : GAMMA * Q[s2 * 4 + a2];  // r = 0 on non-terminal steps
    var delta = target - Q[sa];

    // Replacing traces: zero other actions in s, set e(s,a) = 1.
    for (var i = 0; i < 4; i++) E[s * 4 + i] = 0;
    E[sa] = 1;
    if (!inActive[sa]) { inActive[sa] = 1; active.push(sa); }

    // Update every traced pair, decay its trace, and drop the ones that fade out.
    var decay = GAMMA * LAMBDA, keep = 0;
    for (var j = 0; j < active.length; j++) {
      var idx = active[j];
      Q[idx] += ALPHA * delta * E[idx];
      E[idx] *= decay;
      if (E[idx] >= TRACE_MIN) active[keep++] = idx;
      else { E[idx] = 0; inActive[idx] = 0; }
    }
    active.length = keep;

    steps++;
    visited[s2] = 1;
    s = s2; a = a2;
    return terminal;
  }

  // ---- Rendering ----------------------------------------------------------
  var cellSize = 10, wallPath = null, lastWidth = 0;

  function resize() {
    var w = canvas.parentNode.clientWidth;
    if (w === lastWidth || w === 0) return;  // height changes re-trigger the observer
    lastWidth = w;
    var dpr = window.devicePixelRatio || 1;
    cellSize = w / COLS;
    var h = Math.round(cellSize * ROWS);
    canvas.style.height = h + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildWallPath();
    draw();
  }

  // Each cell draws its north and west walls; the east and south edges of the
  // grid are drawn by the cells along them.
  function buildWallPath() {
    wallPath = new Path2D();
    function seg(x1, y1, x2, y2) { wallPath.moveTo(x1, y1); wallPath.lineTo(x2, y2); }
    for (var i = 0; i < N; i++) {
      var x = colOf(i) * cellSize, y = rowOf(i) * cellSize, w = walls[i];
      var x2 = x + cellSize, y2 = y + cellSize;
      if (w & WALL[NORTH]) seg(x, y, x2, y);
      if (w & WALL[WEST]) seg(x, y, x, y2);
      if ((w & WALL[EAST]) && colOf(i) === COLS - 1) seg(x2, y, x2, y2);
      if ((w & WALL[SOUTH]) && rowOf(i) === ROWS - 1) seg(x, y2, x2, y2);
    }
  }

  function fillCell(i, style, pad) {
    pad = pad || 0;
    ctx.fillStyle = style;
    ctx.fillRect(colOf(i) * cellSize + pad, rowOf(i) * cellSize + pad,
                 cellSize - 2 * pad, cellSize - 2 * pad);
  }

  function rgba(rgb, alpha) { return 'rgba(' + rgb + ',' + alpha.toFixed(3) + ')'; }

  function draw() {
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, cellSize * COLS, cellSize * ROWS);

    // Explored cells get a faint grey; learned value V(s) = max_a Q(s,a)
    // is shown as a green wash that spreads back from the goal. The shade is
    // on a fixed scale (values can't exceed GOAL_REWARD), not relative to the
    // current maximum, so values fading after the goal moves look like fading.
    var i;
    for (i = 0; i < N; i++) {
      if (!visited[i]) continue;
      var v = maxQ(i);
      fillCell(i, v > 0 ? rgba(COLOR.value, 0.06 + 0.3 * Math.min(1, v / GOAL_REWARD)) : COLOR.explored);
    }

    // Eligibility traces, shown as a blue tail behind the agent.
    for (i = 0; i < active.length; i++) {
      fillCell((active[i] / 4) | 0, rgba(COLOR.trace, 0.35 * E[active[i]]));
    }

    fillCell(goal, COLOR.goal, cellSize * 0.15);

    ctx.strokeStyle = COLOR.wall;
    ctx.lineWidth = Math.max(1, cellSize * 0.12);
    ctx.lineCap = 'square';
    ctx.stroke(wallPath);

    ctx.fillStyle = COLOR.agent;
    ctx.beginPath();
    ctx.arc((colOf(s) + 0.5) * cellSize, (rowOf(s) + 0.5) * cellSize, cellSize * 0.33, 0, Math.PI * 2);
    ctx.fill();
  }

  // ---- Caption & controls -------------------------------------------------
  var stats, toggle, speedLinks = [];

  function link(label, onClick) {
    var el = document.createElement('a');
    el.href = '#';
    el.textContent = label;
    el.addEventListener('click', function (ev) { ev.preventDefault(); onClick(); });
    return el;
  }

  // Stats on the first line, controls on their own line below so they stay
  // put as the numbers change width.
  function buildCaption() {
    if (!caption) return;
    stats = document.createTextNode('');
    toggle = link('pause', function () { setRunning(!running); });
    speedLinks = SPEEDS.map(function (sp, i) {
      return link(sp.label, function () { setSpeed(i); });
    });

    var parts = [stats, document.createElement('br'), toggle, ' · ', link('restart', restart), ' · '];
    speedLinks.forEach(function (el, i) { if (i) parts.push(' / '); parts.push(el); });
    parts.push(' · click the maze to move the goal');

    caption.textContent = '';
    parts.forEach(function (p) {
      caption.appendChild(typeof p === 'string' ? document.createTextNode(p) : p);
    });
  }

  function updateCaption() {
    if (!caption) return;
    var text = 'SARSA(λ) agent · episode ' + episode + ' · ' + steps + ' steps';
    if (lastSteps !== null) text += ' · last episode ' + lastSteps + ' · best ' + bestSteps;
    stats.nodeValue = text;
    toggle.textContent = running ? 'pause' : 'play';
    speedLinks.forEach(function (el, i) { el.className = i === speed ? 'is-current' : ''; });
  }

  function render() {
    draw();
    updateCaption();
  }

  // ---- Main loop ----------------------------------------------------------
  // Steps happen at a fixed rate in real time: each frame takes however many
  // steps are due since the last one (frame time capped so a backgrounded tab
  // doesn't come back with a huge burst).
  function frame(now) {
    if (lastTime !== null) pending += Math.min(now - lastTime, MAX_FRAME_MS) / 1000 * SPEEDS[speed].rate;
    lastTime = now;
    var n = Math.floor(pending);
    pending -= n;
    for (var k = 0; k < n; k++) if (step()) endEpisode();
    if (n) render();
    rafId = running ? requestAnimationFrame(frame) : null;
  }

  function setRunning(on) {
    running = on;
    lastTime = null;
    if (on && rafId === null) rafId = requestAnimationFrame(frame);
    updateCaption();
  }

  function setSpeed(i) {
    speed = i;
    updateCaption();
  }

  // ---- Interaction --------------------------------------------------------
  function cellAt(ev) {
    var rect = canvas.getBoundingClientRect();
    var size = rect.width / COLS;
    var r = Math.floor((ev.clientY - rect.top) / size);
    var c = Math.floor((ev.clientX - rect.left) / size);
    return inBounds(r, c) ? r * COLS + c : -1;
  }

  // Click a cell to move the goal there. The agent keeps its Q-values and the
  // current episode carries on: what it learned about the old goal only
  // changes as it collects (or fails to collect) reward under the new one.
  function moveGoal(ev) {
    var g = cellAt(ev);
    if (g < 0 || g === START || g === goal) return;
    goal = g;
    render();
  }

  // New maze, goal back in the corner, agent learning from scratch.
  function newWorld() {
    goal = DEFAULT_GOAL;
    makeMaze();
    resetAgent();
  }

  function restart() {
    newWorld();
    buildWallPath();
    render();
  }

  // ---- Init ---------------------------------------------------------------
  buildCaption();
  canvas.style.cursor = 'pointer';
  canvas.addEventListener('click', moveGoal);

  newWorld();
  resize();
  if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas.parentNode);
  else window.addEventListener('resize', resize);

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  setRunning(!reduceMotion);
})();
