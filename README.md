✈️ F-14 Performance Toolkit — Checkpoint (Sept 20, 2025)

Version: v1.1.3-hotfix8-checkpoint
Scope: Stable UI + perf baseline, with Diagnostics & thrust reconciliation

✅ Changes in this checkpoint

CSV Hardening: Local-first CSV loading from ./data/, with optional GitHub fallback toggle in sidebar.

Diagnostics Expander: Sidebar shows app dir, data dir, and contents for quick debugging of deployment issues.

Thrust/DERATE Reconciliation:

Manual DERATE slider now drives the resolved thrust label & % in Takeoff Results.

Safe wrappers added for _parse_derate_from_label and resolve_thrust_display, so app works regardless of core version.

N1/FF Guidance Table: Uses the resolved thrust mode instead of placeholders.

Calibration Badge: Colored caption “Calibration needed” shown below N1/FF table as a reminder.

📌 Status

UI: Preserved baseline (v1.1.3-hotfix8, perf-optimized, Intersection Margins).

Perf Core: Integrated with f14_perf.csv (NATOPS/DCS hybrid).

Known Pending: Final calibration of engine map; climb & landing integration ongoing in other branches.
