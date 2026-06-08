"""Command-line entry point for PiEEG-agent.

Phase 0 exposes the ingestion spine so you can prove the high-rate intake end
to end against a running PiEEG-server:

    pieeg-server --mock --lsl          # in one terminal (the producer)
    pieeg-agent streams                # list discoverable LSL outlets
    pieeg-agent ingest --seconds 10    # drain the stream into the ring

Later phases add ``run`` (autonomous + copilot agent loop).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import __version__
from .config import PROVIDERS, AgentConfig


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


# ── dispatch ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    if args.command == "streams":
        return cmd_streams(args)
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "config":
        return cmd_config(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
