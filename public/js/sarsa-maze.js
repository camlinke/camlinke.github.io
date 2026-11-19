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
    agent: '#268bd2',  // site link blue
    goal: '#859900',   // solarized green, pairs with the link blue
    trace: '38,139,210',  // agent blue as rgb, for eligibility-trace overlay
    value: '133,153,0',   // goal green as rgb, for learned-value overlay
    explored: '#f9f9f9'   // site's light table-stripe grey
  };

  // Actions: N, E, S, W. Wall bits use the same order.
  var DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];
  var WALL = [1, 2, 4, 8], OPP = [2, 3, 0, 1];

  var START = 0, GOAL = N - 1;

  var walls, Q, E, active, inActive, visited;
  var s, a, episode, steps, lastSteps, bestSteps;
  var running = false, rafId = null, speed = 0, pending = 0, lastTime = null;

  // ---- Maze generation (randomized DFS + a few extra openings) -------------
  function makeMaze() {
    walls = new Uint8Array(N).fill(15);
    var seen = new Uint8Array(N), stack = [START];
    seen[START] = 1;
    while (stack.length) {
      var cur = stack[stack.length - 1];
      var r = (cur / COLS) | 0, c = cur % COLS, opts = [];
      for (var d = 0; d < 4; d++) {
        var nr = r + DR[d], nc = c + DC[d];
        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && !seen[nr * COLS + nc]) opts.push(d);
      }
      if (!opts.length) { stack.pop(); continue; }
      var dir = opts[(Math.random() * opts.length) | 0];
      var nxt = (r + DR[dir]) * COLS + (c + DC[dir]);
      walls[cur] &= ~WALL[dir];
      walls[nxt] &= ~WALL[OPP[dir]];
      seen[nxt] = 1;
      stack.push(nxt);
    }
    var toOpen = Math.floor(N * LOOP_FRACTION);
    while (toOpen > 0) {
      var cell = (Math.random() * N) | 0, dd = (Math.random() * 4) | 0;
      var rr = ((cell / COLS) | 0) + DR[dd], cc = (cell % COLS) + DC[dd];
      if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || !(walls[cell] & WALL[dd])) continue;
      walls[cell] &= ~WALL[dd];
      walls[rr * COLS + cc] &= ~WALL[OPP[dd]];
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
    return Math.random() < EPSILON ? (Math.random() * 4) | 0 : greedy(state);
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
    var terminal = s2 === GOAL;
    var r = terminal ? GOAL_REWARD : 0;
    var a2 = terminal ? 0 : policy(s2);
    var target = terminal ? r : r + GAMMA * Q[s2 * 4 + a2];
    var delta = target - Q[sa];

    // Replacing traces: zero other actions in s, set e(s,a) = 1.
    for (var i = 0; i < 4; i++) E[s * 4 + i] = 0;
    E[sa] = 1;
    if (!inActive[sa]) { inActive[sa] = 1; active.push(sa); }

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
  var cell = 10, wallPath = null, lastWidth = 0;

  function resize() {
    var w = canvas.parentNode.clientWidth;
    var dpr = window.devicePixelRatio || 1;
    if (w === lastWidth || w === 0) return;  // height changes re-trigger the observer
    lastWidth = w;
    cell = w / COLS;
    var h = Math.round(cell * ROWS);
    canvas.style.height = h + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildWallPath();
    draw();
  }

  function buildWallPath() {
    wallPath = new Path2D();
    for (var i = 0; i < N; i++) {
      var x = (i % COLS) * cell, y = ((i / COLS) | 0) * cell, w = walls[i];
      if (w & 1) { wallPath.moveTo(x, y); wallPath.lineTo(x + cell, y); }
      if (w & 8) { wallPath.moveTo(x, y); wallPath.lineTo(x, y + cell); }
      if ((w & 2) && i % COLS === COLS - 1) { wallPath.moveTo(x + cell, y); wallPath.lineTo(x + cell, y + cell); }
      if ((w & 4) && ((i / COLS) | 0) === ROWS - 1) { wallPath.moveTo(x, y + cell); wallPath.lineTo(x + cell, y + cell); }
    }
  }

  function draw() {
    var W = cell * COLS, H = cell * ROWS;
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, W, H);

    // Explored cells get a faint grey; learned value V(s) = max_a Q(s,a)
    // is shown as a green wash that spreads back from the goal. The shade is
    // on a fixed scale (values can't exceed GOAL_REWARD), not relative to the
    // current maximum, so values fading after the goal moves look like fading.
    var i, v, q;
    for (i = 0; i < N; i++) {
      if (!visited[i]) continue;
      v = 0;
      for (q = 0; q < 4; q++) if (Q[i * 4 + q] > v) v = Q[i * 4 + q];
      ctx.fillStyle = v > 0
        ? 'rgba(' + COLOR.value + ',' + (0.06 + 0.3 * Math.min(1, v / GOAL_REWARD)).toFixed(3) + ')'
        : COLOR.explored;
      ctx.fillRect((i % COLS) * cell, ((i / COLS) | 0) * cell, cell, cell);
    }

    // Eligibility traces, shown as a blue tail behind the agent.
    for (i = 0; i < active.length; i++) {
      var st = (active[i] / 4) | 0;
      ctx.fillStyle = 'rgba(' + COLOR.trace + ',' + (0.35 * E[active[i]]).toFixed(3) + ')';
      ctx.fillRect((st % COLS) * cell, ((st / COLS) | 0) * cell, cell, cell);
    }

    // Goal
    var pad = cell * 0.15;
    ctx.fillStyle = COLOR.goal;
    ctx.fillRect((GOAL % COLS) * cell + pad, ((GOAL / COLS) | 0) * cell + pad, cell - 2 * pad, cell - 2 * pad);

    // Walls
    ctx.strokeStyle = COLOR.wall;
    ctx.lineWidth = Math.max(1, cell * 0.12);
    ctx.lineCap = 'square';
    ctx.stroke(wallPath);

    // Agent
    ctx.fillStyle = COLOR.agent;
    ctx.beginPath();
    ctx.arc((s % COLS + 0.5) * cell, (((s / COLS) | 0) + 0.5) * cell, cell * 0.33, 0, Math.PI * 2);
    ctx.fill();
  }

  function updateCaption() {
    if (!caption) return;
    var text = 'SARSA(λ) agent · episode ' + episode + ' · ' + steps + ' steps';
    if (lastSteps !== null) text += ' · last episode ' + lastSteps + ' · best ' + bestSteps;
    caption.firstChild.nodeValue = text;
    toggle.textContent = running ? 'pause' : 'play';
    for (var i = 0; i < speedLinks.length; i++) {
      speedLinks[i].className = i === speed ? 'is-current' : '';
    }
  }

  // ---- Main loop ----------------------------------------------------------
  // Steps happen at a fixed rate in real time: each frame takes however many
  // steps are due since the last one (frame time capped so a backgrounded tab
  // doesn't come back with a huge burst).
  function frame(now) {
    if (lastTime !== null) pending += Math.min(now - lastTime, 250) / 1000 * SPEEDS[speed].rate;
    lastTime = now;
    var n = Math.floor(pending);
    pending -= n;
    for (var k = 0; k < n; k++) {
      if (step()) {
        lastSteps = steps;
        bestSteps = bestSteps === null ? steps : Math.min(bestSteps, steps);
        episode++;
        startEpisode();
      }
    }
    if (n) { draw(); updateCaption(); }
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

  // Click a cell to move the goal there. The agent keeps its Q-values and the
  // current episode carries on: what it learned about the old goal only
  // changes as it collects (or fails to collect) reward under the new one.
  function moveGoal(ev) {
    var rect = canvas.getBoundingClientRect();
    var size = rect.width / COLS;
    var c = Math.floor((ev.clientX - rect.left) / size);
    var r = Math.floor((ev.clientY - rect.top) / size);
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return;
    var g = r * COLS + c;
    if (g === START || g === GOAL) return;
    GOAL = g;
    draw();
    updateCaption();
  }

  function link(label, onClick) {
    var el = document.createElement('a');
    el.href = '#';
    el.textContent = label;
    el.addEventListener('click', function (ev) { ev.preventDefault(); onClick(); });
    return el;
  }

  // New maze, goal back in the corner, agent starts learning from scratch.
  function restart() {
    GOAL = N - 1;
    makeMaze();
    buildWallPath();
    resetAgent();
    draw();
    updateCaption();
  }

  var toggle = link('pause', function () { setRunning(!running); });
  var restartLink = link('restart', restart);
  var speedLinks = SPEEDS.map(function (sp, i) {
    return link(sp.label, function () { setSpeed(i); });
  });

  // Stats on the first line, controls on their own line below so they stay
  // put as the numbers change width.
  if (caption) {
    caption.textContent = '';
    caption.appendChild(document.createTextNode(''));
    caption.appendChild(document.createElement('br'));
    caption.appendChild(toggle);
    caption.appendChild(document.createTextNode(' · '));
    caption.appendChild(restartLink);
    caption.appendChild(document.createTextNode(' · '));
    speedLinks.forEach(function (el, i) {
      if (i) caption.appendChild(document.createTextNode(' / '));
      caption.appendChild(el);
    });
    caption.appendChild(document.createTextNode(' · click the maze to move the goal'));
  }

  canvas.style.cursor = 'pointer';
  canvas.addEventListener('click', moveGoal);

  makeMaze();
  resetAgent();
  resize();
  if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas.parentNode);
  else window.addEventListener('resize', resize);

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  setRunning(!reduceMotion);
})();
