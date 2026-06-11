"""Command-line entry point for PiEEG-agent.

The CLI grows by phase. The ingestion spine proves the high-rate intake, the
perception cascade reduces it to language-sized state, and the copilot reasons
over that state with a provider-agnostic LLM:

    pieeg-server --mock --lsl              # the producer (one terminal)
    pieeg-agent streams                    # list discoverable LSL outlets
    pieeg-agent ingest --seconds 10        # drain the stream into the ring
    pieeg-agent monitor                    # live band powers, state and events
    pieeg-agent ask "am I focused?"        # one-shot brain question
    pieeg-agent chat                       # interactive brain copilot
    pieeg-agent chat --allow-actions       # copilot with gated device control
    pieeg-agent control status             # direct gated server actions

``ask`` / ``chat`` need an LLM provider configured (e.g. $ANTHROPIC_API_KEY).
Device actions (``control`` and ``chat --allow-actions``) talk to PiEEG-server
over its WebSocket control plane and are gated: allowlisted, dry-run by default
for the copilot, cooldown-limited and audited.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .config import PROVIDERS, AgentConfig


def _interactive_llm_setup() -> tuple[str, str]:
    """Prompt user to select a provider and enter API key if needed.
    
    Returns (provider_name, api_key_or_empty).
    """
    print("\n🤖 LLM Provider Setup")
    print("=" * 50)
    print("Choose a provider to power the brain copilot:\n")
    
    # Build menu from PROVIDERS registry
    providers_list = []
    idx = 1
    for key, spec in sorted(PROVIDERS.items()):
        providers_list.append((key, spec))
        label = spec.get("label", key)
        needs_key = spec.get("env_key", "")
        suffix = " (local, no key needed)" if not needs_key else ""
        print(f"  {idx}. {label}{suffix}")
        idx += 1
    
    print()
    
    # Get user choice
    while True:
        try:
            choice = input(f"Select provider [1-{len(providers_list)}]: ").strip()
            if not choice:
                continue
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(providers_list):
                provider_key, provider_spec = providers_list[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(providers_list)}")
        except (ValueError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            sys.exit(1)
    
    provider_label = provider_spec.get("label", provider_key)
    print(f"\n✓ Selected: {provider_label}")
    
    # Get API key if needed
    env_key = provider_spec.get("env_key", "")
    if env_key:
        print(f"\n🔑 API Key Required")
        print(f"This provider needs an API key stored in ${env_key}")
        print("Get yours from:")
        
        # Provider-specific help
        if provider_key == "anthropic":
            print("  → https://console.anthropic.com/settings/keys")
        elif provider_key == "openai":
            print("  → https://platform.openai.com/api-keys")
        elif provider_key == "groq":
            print("  → https://console.groq.com/keys")
        elif provider_key == "together":
            print("  → https://api.together.xyz/settings/api-keys")
        
        print()
        
        try:
            # Use getpass to hide input (API keys are secrets)
            import getpass
            api_key = getpass.getpass(f"Enter your {provider_label} API key: ").strip()
            if not api_key:
                print("\n❌ API key cannot be empty.")
                sys.exit(1)
            print("✓ API key received")
        except (KeyboardInterrupt, EOFError):
            print("\nSetup cancelled.")
            sys.exit(1)
    else:
        api_key = ""
        print("✓ No API key needed (local provider)")
    
    print("\n" + "=" * 50)
    print("Setup complete! Starting agent...\n")
    
    return provider_key, api_key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pieeg-agent",
        description="Provider-agnostic LLM agent for live PiEEG brain activity.",
    )
    parser.add_argument(
        "--version", action="version", version=f"pieeg-agent {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    sub = parser.add_subparsers(dest="command")

    p_streams = sub.add_parser(
        "streams", help="List discoverable LSL streams on the network."
    )
    p_streams.add_argument(
        "--wait", type=float, default=2.0,
        help="Seconds to wait while resolving (default: 2.0).",
    )

    p_ingest = sub.add_parser(
        "ingest", help="Drain an LSL stream into the ring and print live stats."
    )
    p_ingest.add_argument(
        "--name", default=None, help="LSL stream name (default: PiEEG / env)."
    )
    p_ingest.add_argument(
        "--type", dest="stype", default=None,
        help="LSL stream type (default: EEG / env).",
    )
    p_ingest.add_argument(
        "--by", choices=("name", "type"), default=None,
        help="Resolve by stream name or type (default: type / env).",
    )
    p_ingest.add_argument(
        "--seconds", type=float, default=10.0,
        help="How long to ingest; 0 runs until Ctrl+C (default: 10).",
    )
    p_ingest.add_argument(
        "--ring-seconds", type=float, default=None,
        help="Ring-buffer depth in seconds (default: 60 / env).",
    )

    p_monitor = sub.add_parser(
        "monitor",
        help="Run the perception cascade: live band powers, state and events.",
    )
    p_monitor.add_argument(
        "--name", default=None, help="LSL stream name (forces name resolution)."
    )
    p_monitor.add_argument(
        "--type", dest="stype", default=None, help="LSL stream type (default: EEG)."
    )
    p_monitor.add_argument(
        "--by", choices=("name", "type"), default=None,
        help="Resolve by name, or smart-pick the EEG group by type (default).",
    )
    p_monitor.add_argument(
        "--seconds", type=float, default=0.0,
        help="How long to monitor; 0 runs until Ctrl+C (default: 0).",
    )
    p_monitor.add_argument(
        "--ring-seconds", type=float, default=None,
        help="Ring-buffer depth in seconds (default: 60 / env).",
    )
    p_monitor.add_argument(
        "--mains", type=float, default=50.0,
        help="Powerline frequency for the line-noise check (default: 50).",
    )
    p_monitor.add_argument(
        "--feature-hz", type=float, default=8.0,
        help="Feature-extraction rate in Hz (default: 8).",
    )
    p_monitor.add_argument(
        "--state-hz", type=float, default=1.0,
        help="NeuralState emit rate in Hz (default: 1).",
    )
    p_monitor.add_argument(
        "--quiet", action="store_true",
        help="Only print events, not the per-second state line.",
    )

    # ── conversational copilot (Phase 2) ────────────────────────────────
    # Shared perception + LLM flags for ``ask`` and ``chat``.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--name", default=None, help="LSL stream name (forces name resolution)."
    )
    common.add_argument(
        "--type", dest="stype", default=None, help="LSL stream type (default: EEG)."
    )
    common.add_argument(
        "--by", choices=("name", "type"), default=None,
        help="Resolve by name, or smart-pick the EEG group by type (default).",
    )
    common.add_argument(
        "--provider", default=None,
        help=f"LLM provider (default: env / {', '.join(sorted(PROVIDERS))}).",
    )
    common.add_argument(
        "--model", default=None, help="Override the provider's default model."
    )
    common.add_argument(
        "--mains", type=float, default=50.0,
        help="Powerline frequency for the line-noise check (default: 50).",
    )
    common.add_argument(
        "--warmup", type=float, default=3.0,
        help="Seconds to fill the cascade before answering (default: 3).",
    )
    common.add_argument(
        "--ring-seconds", type=float, default=None,
        help="Ring-buffer depth in seconds (default: 60 / env).",
    )
    common.add_argument(
        "--allow-actions", action="store_true",
        help="Give the copilot gated control tools (filter, recording, OSC, "
        "register presets). Off by default \u2014 sessions are read-only.",
    )
    common.add_argument(
        "--execute", action="store_true",
        help="With --allow-actions, actually perform actions instead of just "
        "previewing them (default: dry-run preview only).",
    )
    common.add_argument(
        "--ws-url", default=None,
        help="PiEEG-server control URL (default: ws://localhost:1616 / env).",
    )
    common.add_argument(
        "--token", default=None,
        help="Control-plane auth token, if the server requires one.",
    )
    common.add_argument(
        "--audit-log", default=None,
        help="Where to record gated actions as JSONL "
        "(default: ~/.pieeg-agent/audit.jsonl / $PIEEG_AUDIT_LOG).",
    )
    common.add_argument(
        "--no-audit-log", action="store_true",
        help="Do not persist gated actions to disk (in-memory audit only).",
    )

    p_ask = sub.add_parser(
        "ask", parents=[common],
        help="Ask the brain copilot one question about the live state.",
    )
    p_ask.add_argument("question", help="The question to ask (quote it).")

    sub.add_parser(
        "chat", parents=[common],
        help="Open an interactive chat with the brain copilot.",
    )

    p_web = sub.add_parser(
        "web", parents=[common],
        help="Serve the graphical web UI (streaming chat + live brain panel + "
        "guided pattern training) — same copilot as the CLI.",
    )
    p_web.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose on LAN).",
    )
    p_web.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)."
    )
    p_web.add_argument(
        "--static-dir", default=None,
        help="Built front-end directory to serve (default: frontend/dist if "
        "present).",
    )

    # ── direct gated server control (Phase 3) ───────────────────────────
    # Explicit, human-invoked device actions. Unlike the copilot these run for
    # real by default (you asked); pass --dry-run to preview instead.
    ctl_common = argparse.ArgumentParser(add_help=False)
    ctl_common.add_argument(
        "--ws-url", default=None,
        help="PiEEG-server control URL (default: ws://localhost:1616 / env).",
    )
    ctl_common.add_argument(
        "--token", default=None,
        help="Control-plane auth token, if the server requires one.",
    )
    ctl_common.add_argument(
        "--dry-run", action="store_true",
        help="Preview the action (what would be sent) without performing it.",
    )
    ctl_common.add_argument(
        "--audit-log", default=None,
        help="Where to record gated actions as JSONL "
        "(default: ~/.pieeg-agent/audit.jsonl / $PIEEG_AUDIT_LOG).",
    )
    ctl_common.add_argument(
        "--no-audit-log", action="store_true",
        help="Do not persist gated actions to disk (in-memory audit only).",
    )

    p_control = sub.add_parser(
        "control", help="Send a gated action to PiEEG-server (filter, recording, "
        "OSC, register presets, webhooks) or read the audit log.",
    )
    csub = p_control.add_subparsers(dest="control_cmd")

    csub.add_parser(
        "status", parents=[ctl_common],
        help="Print the server status snapshot (read-only).",
    )

    c_filter = csub.add_parser(
        "set-filter", parents=[ctl_common], help="Enable/disable or retune the "
        "band-pass filter.",
    )
    c_filter.add_argument(
        "--off", action="store_true", help="Disable the filter (default: enable)."
    )
    c_filter.add_argument(
        "--lowcut", type=float, default=1.0, help="High-pass corner Hz (default 1)."
    )
    c_filter.add_argument(
        "--highcut", type=float, default=40.0, help="Low-pass corner Hz (default 40)."
    )

    c_record = csub.add_parser(
        "record", parents=[ctl_common], help="Start or stop server-side recording.",
    )
    c_record.add_argument("action", choices=("start", "stop"))

    c_osc = csub.add_parser(
        "osc", parents=[ctl_common], help="Start or stop the OSC output stream.",
    )
    c_osc.add_argument("action", choices=("start", "stop"))
    c_osc.add_argument("--host", default=None, help="Destination host.")
    c_osc.add_argument("--port", type=int, default=None, help="Destination UDP port.")
    c_osc.add_argument(
        "--mode", choices=("chatbox", "parameters", "both"), default=None,
        help="OSC payload style.",
    )
    c_osc.add_argument(
        "--channel", type=int, default=None,
        help="EEG channel to send (omit for the channel average).",
    )

    c_preset = csub.add_parser(
        "reg-preset", parents=[ctl_common], help="Apply an ADS1299 register preset.",
    )
    c_preset.add_argument(
        "preset",
        choices=("normal", "internal_short", "test_signal", "temp_sensor"),
    )

    csub.add_parser(
        "webhooks", parents=[ctl_common],
        help="List the server's webhook rules (read-only).",
    )

    c_audit = csub.add_parser(
        "audit", parents=[ctl_common],
        help="Show recent gated-action attempts from the audit log (local read).",
    )
    c_audit.add_argument(
        "--limit", type=int, default=20,
        help="Maximum entries to show (default 20).",
    )

    sub.add_parser("config", help="Print the resolved configuration.")
    return parser


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── commands ───────────────────────────────────────────────────────────────


def cmd_streams(args) -> int:
    try:
        from pylsl import resolve_streams
    except Exception as exc:  # pragma: no cover - env issue
        print(f"pylsl unavailable: {exc}", file=sys.stderr)
        return 2

    print(f"Resolving LSL streams ({args.wait:.1f}s)…")
    streams = resolve_streams(args.wait)
    if not streams:
        print("No LSL streams found. Is PiEEG-server running with --lsl?")
        return 1

    print(f"\nFound {len(streams)} stream(s):\n")
    header = f"{'NAME':<20} {'TYPE':<8} {'CH':>3} {'RATE':>8}  SOURCE"
    print(header)
    print("-" * len(header))
    for s in streams:
        srate = s.nominal_srate()
        rate = f"{srate:.0f}Hz" if srate > 0 else "irim"
        print(
            f"{s.name():<20} {s.type():<8} {s.channel_count():>3} "
            f"{rate:>8}  {s.source_id()}"
        )
    return 0


def cmd_ingest(args) -> int:
    from .ingest import LSLInlet, LSLStreamConfig

    cfg = AgentConfig.from_env(
        lsl_name=args.name,
        lsl_type=args.stype,
        lsl_resolve_by=args.by,
        ring_seconds=args.ring_seconds,
    )

    inlet = LSLInlet(
        LSLStreamConfig(
            name=cfg.lsl_name,
            stype=cfg.lsl_type,
            resolve_by=cfg.lsl_resolve_by,
            resolve_timeout=cfg.lsl_resolve_timeout,
            ring_seconds=cfg.ring_seconds,
        )
    )

    target = cfg.lsl_name if cfg.lsl_resolve_by == "name" else cfg.lsl_type
    print(f"Resolving LSL stream by {cfg.lsl_resolve_by}={target!r}…")
    if not inlet.resolve():
        print(
            "Could not find the stream. Start the producer with:\n"
            "  pieeg-server --mock --lsl",
            file=sys.stderr,
        )
        return 1

    labels = inlet.channel_labels
    preview = ", ".join(labels[:8]) + (" …" if len(labels) > 8 else "")
    print(
        f"Connected to {inlet.stream_name!r}: {inlet.num_channels} ch @ "
        f"{inlet.sample_rate:.0f} Hz\n  channels: {preview}\n"
    )

    inlet.start()
    deadline = None if args.seconds <= 0 else time.monotonic() + args.seconds
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(1.0)
            _print_status(inlet)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        inlet.stop()

    st = inlet.stats()
    print(
        f"\n\nDone. Ingested {st['samples']} samples across "
        f"{st['channels']} channels.\n"
        f"  nominal rate : {st['nominal_srate']:.0f} Hz\n"
        f"  delivered    : ~{st['effective_rate']:.0f}/s "
        f"(producer-bound; the mock under-delivers on Windows timers)\n"
        f"  backlog      : {st['queued']} samples queued at exit\n"
        f"  reconnects   : {st['lost_count']}"
    )
    # "Keeping up" is about the consumer, not the producer's clock: we want
    # continuous intake, zero losses, and no growing inlet backlog.
    keeping_up = (
        st["samples"] > 0
        and st["lost_count"] == 0
        and 0 <= st["queued"] < max(st["nominal_srate"], 1.0)
    )
    if keeping_up:
        print("  status       : OK \u2014 consumer kept up with the stream.")
        return 0
    print(
        "  status       : FELL BEHIND \u2014 consumer did not keep up.",
        file=sys.stderr,
    )
    return 1


def cmd_config(args) -> int:
    cfg = AgentConfig.from_env()
    print("Resolved configuration:\n")
    print(f"  LSL stream      : {cfg.lsl_resolve_by}={cfg.lsl_name!r}/{cfg.lsl_type!r}")
    print(f"  Ring depth      : {cfg.ring_seconds:.0f}s")
    print(f"  LLM provider    : {cfg.provider} ({cfg.provider_spec.get('label', '?')})")
    print(f"  LLM model       : {cfg.model}")
    print(f"  API key present : {cfg.has_api_key}")
    print(f"  Server WS       : {cfg.ws_url}")
    print(f"  Known providers : {', '.join(sorted(PROVIDERS))}")
    problems = cfg.validate()
    if problems:
        print("\n  Notes:")
        for p in problems:
            print(f"    - {p}")
    return 0


def _print_status(inlet) -> None:
    st = inlet.stats()
    ring = inlet.ring
    peek = ""
    if ring is not None:
        data, _ = ring.latest(1)
        if data.shape[0]:
            vals = ", ".join(f"{v:+6.1f}" for v in data[0, : min(4, data.shape[1])])
            peek = f"  µV[{vals}]"
    fill_pct = (
        100.0 * st["ring_fill"] / st["ring_capacity"] if st["ring_capacity"] else 0.0
    )
    age = f"{st['staleness'] * 1000:5.1f}ms" if st["staleness"] is not None else "  n/a"
    line = (
        f"\r  {st['recent_rate']:6.1f}/s | "
        f"{st['samples']:>8} samp | "
        f"queued {st['queued']:>4} | "
        f"ring {fill_pct:5.1f}% | "
        f"age {age}{peek}   "
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def cmd_monitor(args) -> int:
    from .ingest import (
        LSLInlet,
        LSLStreamConfig,
        discover_streams,
        rank_eeg_streams,
    )
    from .perceive import CascadeConfig, PerceptionCascade

    cfg = AgentConfig.from_env(
        lsl_name=args.name,
        lsl_type=args.stype,
        lsl_resolve_by=args.by,
        ring_seconds=args.ring_seconds,
    )
    inlet = LSLInlet(
        LSLStreamConfig(
            name=cfg.lsl_name,
            stype=cfg.lsl_type,
            resolve_by=cfg.lsl_resolve_by,
            resolve_timeout=cfg.lsl_resolve_timeout,
            ring_seconds=cfg.ring_seconds,
        )
    )

    # Explicit --name forces a single unambiguous outlet; otherwise resolve by
    # type and smart-pick the brain-EEG group out of any multi-group profile
    # (EEG_PiEEG vs EOG_PiEEG / AUX_PiEEG, which all advertise type 'EEG').
    connected = False
    if cfg.lsl_resolve_by == "name":
        print(f"Resolving LSL stream by name={cfg.lsl_name!r}\u2026")
        connected = inlet.resolve()
    else:
        print(f"Discovering EEG streams ({cfg.lsl_resolve_timeout:.1f}s)\u2026")
        ranked = rank_eeg_streams(
            discover_streams(cfg.lsl_resolve_timeout), cfg.lsl_type
        )
        if ranked:
            _print_stream_menu(ranked)
            connected = inlet.connect_info(ranked[0])
    if not connected:
        print(
            "Could not find an EEG stream. Start the producer with:\n"
            "  pieeg-server --mock --lsl",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nMonitoring {inlet.stream_name!r}: {inlet.num_channels} ch @ "
        f"{inlet.sample_rate:.0f} Hz  (mains {args.mains:.0f} Hz)\n"
        + "-" * 78
    )

    inlet.start()
    cascade = PerceptionCascade(
        inlet,
        CascadeConfig(
            mains_hz=args.mains,
            feature_hz=args.feature_hz,
            state_hz=args.state_hz,
        ),
        on_state=None if args.quiet else _print_state_line,
        on_event=_print_event_line,
        on_artifact=_print_artifact_line,
    )
    cascade.start()

    deadline = None if args.seconds <= 0 else time.monotonic() + args.seconds
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cascade.stop()
        inlet.stop()

    st = cascade.stats()
    ist = inlet.stats()
    print(
        f"\nDone. {st['features']} feature frames \u2192 {st['states']} states "
        f"\u2192 {st['events']} events, from {ist['samples']} samples "
        f"({ist['lost_count']} reconnects)."
    )
    return 0 if st["states"] > 0 else 1


def _connect_eeg_inlet(cfg):
    """Resolve (by name) or discover+smart-pick (by type) the EEG inlet.

    Shared by ``monitor``, ``ask`` and ``chat``. Returns a connected, *not yet
    started* :class:`LSLInlet`, or ``None`` if no stream was found.
    """
    from .ingest import (
        LSLInlet,
        LSLStreamConfig,
        discover_streams,
        rank_eeg_streams,
    )

    inlet = LSLInlet(
        LSLStreamConfig(
            name=cfg.lsl_name,
            stype=cfg.lsl_type,
            resolve_by=cfg.lsl_resolve_by,
            resolve_timeout=cfg.lsl_resolve_timeout,
            ring_seconds=cfg.ring_seconds,
        )
    )
    if cfg.lsl_resolve_by == "name":
        print(f"Resolving LSL stream by name={cfg.lsl_name!r}\u2026")
        return inlet if inlet.resolve() else None

    print(f"Discovering EEG streams ({cfg.lsl_resolve_timeout:.1f}s)\u2026")
    ranked = rank_eeg_streams(discover_streams(cfg.lsl_resolve_timeout), cfg.lsl_type)
    if not ranked:
        return None
    _print_stream_menu(ranked)
    return inlet if inlet.connect_info(ranked[0]) else None


@dataclass
class _CopilotSession:
    """The assembled live stack shared by ``ask`` / ``chat`` / ``web``."""

    copilot: object
    inlet: object
    cascade: object
    client: object
    senses: object
    decode: object
    cfg: object
    actions: object = None  # ServerActions when --allow-actions is set


def _start_copilot(args):
    """Bring up inlet + cascade + copilot for the ``ask`` / ``chat`` / ``web``
    commands.

    Returns a :class:`_CopilotSession` on success or ``None`` after printing an
    actionable error (missing stream, unconfigured provider, unreachable
    control plane). ``client`` is the server control connection when
    ``--allow-actions`` is set, otherwise ``None``; ``senses`` / ``decode`` are
    exposed so the web layer can read the same live state the copilot does.
    """
    from .agent import (
        ACTUATOR_SYSTEM_PROMPT,
        SAFE_ACTIONS,
        SYSTEM_PROMPT,
        CombinedToolset,
        Copilot,
        DecodeTools,
        NeuralTools,
    )
    from .llm import ProviderError, get_provider
    from .perceive import CascadeConfig, PerceptionCascade

    cfg = AgentConfig.from_env(
        lsl_name=args.name,
        lsl_type=args.stype,
        lsl_resolve_by=args.by,
        provider=args.provider,
        model=args.model,
        ring_seconds=args.ring_seconds,
    )

    # Fail fast on LLM config *before* touching hardware, so the user isn't
    # told to plug in a headset only to hit a missing-key error afterwards.
    # If provider is not configured, prompt interactively.
    provider = None
    try:
        provider = get_provider(cfg)
    except ProviderError as exc:
        # Check if this is a missing API key error and if we're in an interactive terminal
        if sys.stdin.isatty() and "API key" in str(exc):
            # Interactive setup
            provider_name, api_key = _interactive_llm_setup()
            
            # Set environment variables for this session
            if api_key:
                spec = PROVIDERS.get(provider_name, {})
                env_key = spec.get("env_key", "")
                if env_key:
                    os.environ[env_key] = api_key
            
            # Rebuild config with the selected provider
            cfg = AgentConfig.from_env(
                lsl_name=args.name,
                lsl_type=args.stype,
                lsl_resolve_by=args.by,
                provider=provider_name,
                model=args.model,
                ring_seconds=args.ring_seconds,
            )
            
            # Retry provider creation
            try:
                provider = get_provider(cfg)
            except ProviderError as retry_exc:
                print(f"LLM provider setup failed: {retry_exc}", file=sys.stderr)
                return None
        else:
            # Non-interactive or different error - fail with original message
            print(f"LLM provider not ready: {exc}", file=sys.stderr)
            print("\nTip: Run this command interactively to configure the provider,", file=sys.stderr)
            print("     or set the required environment variables.", file=sys.stderr)
            return None

    # Optionally bring up the gated actuator side (also before hardware, so a
    # bad control URL fails fast). Default sessions stay read-only.
    client = None
    actuator = None
    allow_actions = getattr(args, "allow_actions", False)
    if allow_actions:
        actuator, client = _build_actuator(args, cfg, SAFE_ACTIONS)
        if actuator is None:
            return None

    inlet = _connect_eeg_inlet(cfg)
    if inlet is None:
        if client is not None:
            client.close()
        print(
            "Could not find an EEG stream. Start the producer with:\n"
            "  pieeg-server --mock --lsl",
            file=sys.stderr,
        )
        return None

    mode = "control" if allow_actions else "read-only"
    print(
        f"\nCopilot ready on {inlet.stream_name!r}: {inlet.num_channels} ch @ "
        f"{inlet.sample_rate:.0f} Hz  \u2014  {cfg.provider}:{cfg.model}  "
        f"({mode})"
    )

    inlet.start()
    cascade = PerceptionCascade(inlet, CascadeConfig(mains_hz=args.mains))
    senses = NeuralTools(cascade)
    decode = DecodeTools(cascade)
    # Drive the live pattern bank (and any in-progress training capture) from
    # every cascade frame. Wired here because cascade and decoder reference
    # each other.
    cascade.set_on_frame(decode.on_frame)
    cascade.start()

    # Let the cascade fill its first analysis window so early questions have a
    # real state to read instead of "still warming up".
    if args.warmup > 0:
        print(f"Warming up ({args.warmup:.0f}s)\u2026")
        _wait_for_state(cascade, args.warmup)

    if actuator is not None:
        tools = CombinedToolset(senses, decode, actuator)
        copilot = Copilot(provider, tools, system=ACTUATOR_SYSTEM_PROMPT)
        # Build actions for direct web control (reuses the same client/gate)
        from .server import ActionGate, ActionPolicy, AuditLog, ServerActions
        audit_path = (
            None if getattr(args, "no_audit_log", False) else
            _audit_log_path(args)
        )
        gate = ActionGate(
            policy=ActionPolicy(allowed=SAFE_ACTIONS),
            audit=AuditLog(audit_path),
            dry_run=not getattr(args, "execute", False),
        )
        actions = ServerActions(client, gate)
    else:
        tools = CombinedToolset(senses, decode)
        copilot = Copilot(provider, tools, system=SYSTEM_PROMPT)
        actions = None
    return _CopilotSession(
        copilot=copilot,
        inlet=inlet,
        cascade=cascade,
        client=client,
        senses=senses,
        decode=decode,
        cfg=cfg,
        actions=actions,
    )


def _build_actuator(args, cfg, safe_actions):
    """Connect the control plane and build gated actuator tools.

    Returns ``(ActuatorTools, ServerControlClient)`` or ``(None, None)`` after
    printing an error. The policy allows only the safe action set; dry-run is
    the default unless ``--execute`` was passed.
    """
    from .agent import ActuatorTools
    from .server import (
        ActionGate,
        ActionPolicy,
        ServerActions,
        ServerControlClient,
        ServerControlError,
    )

    ws_url = args.ws_url or cfg.ws_url
    token = _control_token(args)
    execute = getattr(args, "execute", False)
    client = ServerControlClient(ws_url, token=token)
    try:
        welcome = client.connect()
    except ServerControlError as exc:
        print(
            f"Could not reach the control plane at {ws_url}: {exc}\n"
            "Start the server with:  pieeg-server --mock --lsl",
            file=sys.stderr,
        )
        return None, None

    policy = ActionPolicy.allow(*safe_actions, dry_run=not execute, cooldown_s=3.0)
    gate = ActionGate(policy, _audit_log(args))
    actions = ServerActions(client, gate)
    verb = "EXECUTE" if execute else "dry-run preview"
    mock = " (mock)" if welcome.get("mock") else ""
    print(f"Control plane connected at {ws_url}{mock}  \u2014  actions: {verb}")
    if gate.audit.path:
        print(f"Audit log: {gate.audit.path}")
    return ActuatorTools(actions), client


def _control_token(args):
    return args.token or os.environ.get("PIEEG_WS_TOKEN")


def _audit_log_path(args) -> str | None:
    """Resolve the audit-log JSONL path, or ``None`` if persistence is off.

    Order: ``--no-audit-log`` wins; else ``--audit-log`` flag, then
    ``$PIEEG_AUDIT_LOG``, then a default under the user's home.
    """
    if getattr(args, "no_audit_log", False):
        return None
    return (
        getattr(args, "audit_log", None)
        or os.environ.get("PIEEG_AUDIT_LOG")
        or str(Path.home() / ".pieeg-agent" / "audit.jsonl")
    )


def _audit_log(args):
    """Build an :class:`AuditLog`, persisting to JSONL unless disabled.

    Creates the parent directory if needed; falls back to an in-memory log if
    the path can't be prepared, so a bad path never blocks an action.
    """
    from .server import AuditLog

    path = _audit_log_path(args)
    if path is None:
        return AuditLog()
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"Warning: cannot prepare audit log {path}: {exc} "
            "(continuing without persistence).",
            file=sys.stderr,
        )
        return AuditLog()
    return AuditLog(path=path)


def _wait_for_state(cascade, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cascade.latest_state() is not None:
            return
        time.sleep(0.1)


def cmd_ask(args) -> int:
    started = _start_copilot(args)
    if started is None:
        return 1
    copilot, inlet, cascade, client = (
        started.copilot, started.inlet, started.cascade, started.client
    )
    try:
        result = copilot.ask(args.question)
    except Exception as exc:
        print(f"\nCopilot error: {exc}", file=sys.stderr)
        return 1
    finally:
        cascade.stop()
        inlet.stop()
        if client is not None:
            client.close()

    if result.tool_calls:
        print(f"  (consulted: {', '.join(result.tool_calls)})")
    print("\n" + result.text)
    return 0


def cmd_chat(args) -> int:
    started = _start_copilot(args)
    if started is None:
        return 1
    copilot, inlet, cascade, client = (
        started.copilot, started.inlet, started.cascade, started.client
    )
    print(
        "\nChatting with PiEEG Copilot. Ask about focus, relaxation, signal "
        "quality\u2026\nType 'exit' (or Ctrl+D) to quit.\n"
    )
    try:
        while True:
            try:
                question = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", ":q"):
                break
            try:
                result = copilot.ask(question)
            except Exception as exc:
                print(f"  copilot error: {exc}", file=sys.stderr)
                continue
            if result.tool_calls:
                print(f"  (consulted: {', '.join(result.tool_calls)})")
            print(f"\ncopilot > {result.text}\n")
    finally:
        cascade.stop()
        inlet.stop()
        if client is not None:
            client.close()
    print("Bye.")
    return 0


def _default_static_dir() -> str | None:
    """The built front-end directory (``frontend/dist``), if it exists.

    Looks next to the repo root so a plain ``pieeg-agent web`` serves the UI
    after ``npm run build`` with no flags. Returns ``None`` when unbuilt, in
    which case the API still runs (use the Vite dev server in development).
    """
    candidate = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    return str(candidate) if candidate.is_dir() else None


def cmd_web(args) -> int:
    """Serve the graphical web UI over the same copilot + cascade as the CLI."""
    try:
        import uvicorn
    except ImportError:
        print(
            "The web UI needs FastAPI/uvicorn. Install the extra:\n"
            "  pip install 'pieeg-agent[web]'",
            file=sys.stderr,
        )
        return 2

    started = _start_copilot(args)
    if started is None:
        return 1

    from .web import WebEngine, create_app

    inlet = started.inlet
    info = {
        "stream": inlet.stream_name,
        "channels": inlet.num_channels,
        "rate": round(inlet.sample_rate, 1),
        "provider": started.cfg.provider,
        "model": started.cfg.model,
        "control": started.client is not None,
    }
    engine = WebEngine(
        copilot=started.copilot,
        senses=started.senses,
        decode=started.decode,
        info=info,
        actions=started.actions,
    )
    static_dir = args.static_dir or _default_static_dir()
    app = create_app(engine, static_dir=static_dir)

    url = f"http://{args.host}:{args.port}"
    print(f"\nPiEEG Agent web UI on {url}   (Ctrl+C to stop)")
    if static_dir and Path(static_dir).is_dir():
        print(f"  serving front-end from {static_dir}")
    else:
        print("  front-end not built \u2014 API live at /api and /ws.")
        print("  build it:  cd frontend && npm install && npm run build")
        print("  or dev:    cd frontend && npm run dev   (proxies to this API)")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        started.cascade.stop()
        inlet.stop()
        if started.client is not None:
            started.client.close()
    return 0


def cmd_control(args) -> int:
    """Send one explicit, gated action to PiEEG-server (or read its status)."""
    from .server import (
        ActionGate,
        ActionPolicy,
        AuditLog,
        ServerActions,
        ServerControlClient,
        ServerControlError,
    )

    if not getattr(args, "control_cmd", None):
        print(
            "Specify a control subcommand: status, set-filter, record, osc, "
            "reg-preset, webhooks, audit.",
            file=sys.stderr,
        )
        return 2

    # `audit` is a purely local read of the JSONL log — no server connection.
    if args.control_cmd == "audit":
        return _cmd_control_audit(args)

    action_name, invoke = _resolve_control(args)
    if invoke is None:
        print(f"Unknown control command: {args.control_cmd}", file=sys.stderr)
        return 2

    cfg = AgentConfig.from_env()
    ws_url = args.ws_url or cfg.ws_url
    client = ServerControlClient(ws_url, token=_control_token(args))
    try:
        welcome = client.connect()
    except ServerControlError as exc:
        print(
            f"Could not reach the control plane at {ws_url}: {exc}\n"
            "Start the server with:  pieeg-server --mock --lsl",
            file=sys.stderr,
        )
        return 1

    mock = " (mock)" if welcome.get("mock") else ""
    print(
        f"Connected to {ws_url}{mock}: {welcome.get('channels', '?')} ch @ "
        f"{welcome.get('sample_rate', '?')} Hz"
    )

    # Explicit human commands run for real by default; --dry-run previews. A
    # read (action_name is None) needs no gate, and isn't audited; a real
    # action is allowlisted to just itself and recorded to the audit log.
    if action_name is None:
        policy = ActionPolicy()
        audit = AuditLog()
    else:
        policy = ActionPolicy.allow(
            action_name, dry_run=getattr(args, "dry_run", False), cooldown_s=0.0
        )
        audit = _audit_log(args)
    actions = ServerActions(client, ActionGate(policy, audit))

    try:
        result = invoke(actions)
    except ServerControlError as exc:
        print(f"  control error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    _print_control_result(action_name, result)
    return 0


def _resolve_control(args):
    """Map a control subcommand to ``(action_name, fn)``.

    ``action_name`` is ``None`` for read-only commands (no gating); otherwise
    it is the gated action name. ``fn`` takes a :class:`ServerActions` and
    returns the result/envelope to print.
    """
    cmd = args.control_cmd
    if cmd == "status":
        return None, lambda a: a.server_info()
    if cmd == "webhooks":
        return None, lambda a: a.list_webhooks()
    if cmd == "set-filter":
        return "set_filter", lambda a: a.set_filter(
            enabled=not args.off, lowcut=args.lowcut, highcut=args.highcut
        )
    if cmd == "record":
        if args.action == "start":
            return "start_record", lambda a: a.start_recording()
        return "stop_record", lambda a: a.stop_recording()
    if cmd == "osc":
        if args.action == "start":
            osc_cfg: dict = {}
            for key in ("host", "mode"):
                if getattr(args, key) is not None:
                    osc_cfg[key] = getattr(args, key)
            if args.port is not None:
                osc_cfg["port"] = args.port
            if args.channel is not None:
                osc_cfg["channel"] = args.channel
            return "osc_start", lambda a: a.start_osc(osc_cfg)
        return "osc_stop", lambda a: a.stop_osc()
    if cmd == "reg-preset":
        return "reg_preset", lambda a: a.apply_register_preset(args.preset)
    return None, None


_CONTROL_MARKERS = {
    "executed": "OK",
    "dry_run": "DRY-RUN",
    "denied": "DENIED",
    "error": "ERROR",
}


def _print_control_result(action_name, result) -> None:
    import json

    if action_name is None:  # read-only command
        print(json.dumps(result, indent=2))
        return

    outcome = result.get("outcome", "?")
    marker = _CONTROL_MARKERS.get(outcome, outcome.upper())
    reason = result.get("reason", "")
    print(f"\n[{marker}] {action_name}" + (f"  -  {reason}" if reason else ""))
    if outcome == "dry_run":
        print(f"  would send: {json.dumps(result.get('would_send', {}))}")
    elif outcome == "executed":
        print(f"  result: {json.dumps(result.get('result', {}))}")


def _cmd_control_audit(args) -> int:
    """Print the most recent gated-action attempts from the audit log."""
    import json

    path = _audit_log_path(args)
    if path is None:
        print("Audit logging is disabled (--no-audit-log).", file=sys.stderr)
        return 1

    p = Path(path)
    if not p.exists():
        print(f"No audit log yet at {path}.")
        print(
            "Actions you run (control … or chat --allow-actions --execute) "
            "will be recorded there."
        )
        return 0

    try:
        raw = p.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"Could not read audit log {path}: {exc}", file=sys.stderr)
        return 1

    entries = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:  # skip a torn/partial line
            continue

    if not entries:
        print(f"Audit log {path} is empty.")
        return 0

    limit = getattr(args, "limit", 20) or 20
    shown = entries[-limit:]
    noun = "entry" if len(entries) == 1 else "entries"
    print(f"Audit log: {path}  ({len(entries)} {noun}, showing {len(shown)})\n")
    for e in shown:
        ts = e.get("timestamp")
        when = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            if isinstance(ts, (int, float)) else "?"
        )
        marker = _CONTROL_MARKERS.get(e.get("outcome", ""), str(e.get("outcome")))
        line = f"  {when}  [{marker:>7}] {e.get('action', '?')}"
        reason = e.get("reason", "")
        if reason:
            line += f"  -  {reason}"
        print(line)
    return 0


def _print_stream_menu(ranked) -> None:
    print(f"Found {len(ranked)} EEG-type stream(s):")
    for i, s in enumerate(ranked):
        marker = ">" if i == 0 else " "
        sel = "  (selected)" if i == 0 else ""
        print(
            f"  {marker} {s.name():<16} {s.channel_count():>2} ch @ "
            f"{s.nominal_srate():.0f} Hz{sel}"
        )


def _band_str(rel_bands: dict) -> str:
    from .perceive import BAND_NAMES

    return " ".join(f"{b[0].lower()}{rel_bands.get(b, 0.0):.2f}" for b in BAND_NAMES)


def _print_state_line(state) -> None:
    ts = time.strftime("%H:%M:%S", time.localtime(state.timestamp))
    tag = "~" if state.warming_up else " "
    qnote = (
        "clean" if not state.bad_channels
        else "check " + ",".join(state.bad_channels)
    )
    print(
        f"{ts}{tag}| focus {state.focus:.2f} relax {state.relax:.2f} "
        f"engage {state.engagement:.2f} | {_band_str(state.rel_bands)} "
        f"| dom {state.dominant_band:<5} | Q {state.signal_quality:.2f} {qnote}"
    )


def _print_event_line(event) -> None:
    ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
    mark = "!!" if event.severity == "warn" else ">>"
    print(f"   {mark} {ts}  {event.type}  -  {event.detail}")


def _print_artifact_line(art) -> None:
    ts = time.strftime("%H:%M:%S", time.localtime(art.timestamp))
    print(
        f"   ·· {ts}  {art.type}  -  {art.detail} "
        f"({art.confidence:.0%} conf)"
    )


# ── dispatch ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    if args.command == "streams":
        return cmd_streams(args)
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "monitor":
        return cmd_monitor(args)
    if args.command == "ask":
        return cmd_ask(args)
    if args.command == "chat":
        return cmd_chat(args)
    if args.command == "web":
        return cmd_web(args)
    if args.command == "control":
        return cmd_control(args)
    if args.command == "config":
        return cmd_config(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
