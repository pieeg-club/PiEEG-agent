import { memo, useEffect, useRef } from "react";
import type { CascadeStats, NeuralState, Events } from "../types";

/**
 * Cascade Monitor — real-time visualization of the perception pipeline.
 * 
 * Shows the ring → features → state → events flow with processing stats,
 * index sparklines, and threshold markers. This is the "behind the scenes"
 * animation of how Schmitt triggers and the cascade work.
 */
export const CascadeMonitor = memo(function CascadeMonitor({
  cascade,
  state,
  events,
}: {
  cascade?: CascadeStats;
  state?: NeuralState;
  events?: Events;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const historyRef = useRef<{
    focus: number[];
    relax: number[];
    engagement: number[];
  }>({
    focus: [],
    relax: [],
    engagement: [],
  });
  const needsRenderRef = useRef(false);

  // Update sparkline history
  useEffect(() => {
    if (!state || state.status === "no_data") return;
    const hist = historyRef.current;
    const maxPoints = 60; // 60 seconds @ 1Hz

    if (state.focus != null) {
      hist.focus.push(state.focus);
      if (hist.focus.length > maxPoints) hist.focus.shift();
    }
    if (state.relax != null) {
      hist.relax.push(state.relax);
      if (hist.relax.length > maxPoints) hist.relax.shift();
    }
    if (state.engagement != null) {
      hist.engagement.push(state.engagement);
      if (hist.engagement.length > maxPoints) hist.engagement.shift();
    }
    
    // Mark that we need a render, but don't trigger immediately
    needsRenderRef.current = true;
  }, [state]);

  // Render sparklines using RAF loop for smooth 60fps updates
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let resizeObserver: ResizeObserver | null = null;
    let canvasWidth = 0;
    let canvasHeight = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.offsetWidth;
      const h = canvas.offsetHeight;
      if (w === canvasWidth && h === canvasHeight) return;
      canvasWidth = w;
      canvasHeight = h;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      // Note: setting canvas.width/height resets the transformation matrix
      // We apply scale in the render loop instead
      needsRenderRef.current = true;
    };

    const render = () => {
      // Always continue the loop
      rafRef.current = requestAnimationFrame(render);
      
      if (!needsRenderRef.current) {
        return;
      }
      needsRenderRef.current = false;

      const dpr = window.devicePixelRatio || 1;
      const w = canvasWidth;
      const h = canvasHeight;
      
      // Skip rendering if canvas has no size yet
      if (w === 0 || h === 0) return;
      
      // Reset and apply scale transformation
      ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset to identity
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      const hist = historyRef.current;
      const rowH = h / 3;
      const hi = 0.70;
      const lo = 0.30;

      // Helper to draw one sparkline with thresholds
      const drawLine = (data: number[], y: number, color: string, label: string) => {
        if (data.length < 2) return;

        const x0 = 10;
        const x1 = w - 10;
        const dx = (x1 - x0) / Math.max(data.length - 1, 1);

        // Threshold lines
        ctx.strokeStyle = "#444";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(x0, y + rowH * (1 - hi));
        ctx.lineTo(x1, y + rowH * (1 - hi));
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x0, y + rowH * (1 - lo));
        ctx.lineTo(x1, y + rowH * (1 - lo));
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = "#888";
        ctx.font = "10px monospace";
        ctx.fillText(label, 2, y + 10);

        // Sparkline
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        data.forEach((val, i) => {
          const x = x0 + i * dx;
          const py = y + rowH * (1 - val);
          if (i === 0) ctx.moveTo(x, py);
          else ctx.lineTo(x, py);
        });
        ctx.stroke();

        // Current value marker
        const last = data[data.length - 1];
        const lastX = x0 + (data.length - 1) * dx;
        const lastY = y + rowH * (1 - last);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(lastX, lastY, 2, 0, Math.PI * 2);
        ctx.fill();

        // Value text
        ctx.fillStyle = color;
        ctx.font = "bold 11px monospace";
        ctx.fillText(last.toFixed(2), x1 - 30, y + 10);
      };

      drawLine(hist.focus, 0, "#4fc3f7", "FOC");
      drawLine(hist.relax, rowH, "#81c784", "REL");
      drawLine(hist.engagement, rowH * 2, "#ffb74d", "ENG");
    };

    resize();
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    rafRef.current = requestAnimationFrame(render);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (resizeObserver) resizeObserver.disconnect();
    };
  }, []);

  if (!cascade) {
    return (
      <section className="card cascade-card">
        <div className="card-title">Cascade</div>
        <div className="cascade-waiting">waiting for data…</div>
      </section>
    );
  }

  const lastEvent = events?.events?.[events.events.length - 1];
  const eventAge = lastEvent
    ? Math.floor(Date.now() / 1000 - lastEvent.timestamp)
    : null;

  // Calculate rates
  const featureHz = cascade.ticks > 0 ? (cascade.features / cascade.ticks) * 8 : 0;
  const stateHz = cascade.features > 0 ? (cascade.states / cascade.features) * 8 : 0;

  return (
    <section className="card cascade-card">
      <div className="card-title">Cascade Monitor</div>
      
      <div className="cascade-flow">
        <div className="flow-stage">
          <div className="flow-label">RAW</div>
          <div className="flow-rate">250Hz</div>
        </div>
        <div className="flow-arrow">━━━&gt;</div>
        <div className="flow-stage active">
          <div className="flow-label">FEAT</div>
          <div className="flow-rate">{featureHz.toFixed(1)}Hz</div>
        </div>
        <div className="flow-arrow">━━━&gt;</div>
        <div className="flow-stage active">
          <div className="flow-label">STATE</div>
          <div className="flow-rate">{stateHz.toFixed(1)}Hz</div>
        </div>
        <div className="flow-arrow">━━━&gt;</div>
        <div className="flow-stage">
          <div className="flow-label">EVENT</div>
          <div className="flow-rate">debounce</div>
        </div>
      </div>

      <div className="cascade-stats">
        <div className="stat-row">
          <span className="stat-label">Ticks:</span>
          <span className="stat-value">{cascade.ticks.toLocaleString()}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Features:</span>
          <span className="stat-value">{cascade.features.toLocaleString()}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">States:</span>
          <span className="stat-value">{cascade.states.toLocaleString()}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Events:</span>
          <span className="stat-value">{cascade.events}</span>
        </div>
      </div>

      <canvas ref={canvasRef} className="cascade-sparklines" />

      {lastEvent && (
        <div className="cascade-last-event">
          <span className="event-type">{lastEvent.type.replace(/_/g, " ")}</span>
          {eventAge != null && (
            <span className="event-age">{eventAge}s ago</span>
          )}
        </div>
      )}
    </section>
  );
});
