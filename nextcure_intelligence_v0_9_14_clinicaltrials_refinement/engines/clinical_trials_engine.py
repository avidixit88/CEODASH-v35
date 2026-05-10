"""Live ClinicalTrials.gov intelligence lane.

Phase 1 goal:
- Pull compact live trial signals from ClinicalTrials.gov on each analysis run.
- Score and synthesize them into the four executive buckets.
- Preserve backend hooks so the same normalized records can later be persisted
  into a database without changing the executive UI contract.

This module intentionally avoids a Streamlit cache. While the prototype is on
Streamlit Community Cloud, each run fetches fresh data with small page sizes and
short timeouts, then fails gracefully if the upstream service is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from config.clinical_trials_sources import (
    CLINICAL_TRIALS_PAGE_SIZE,
    CLINICAL_TRIALS_TIMEOUT_SECONDS,
    CLINICAL_TRIAL_SEARCH_SPECS,
    ClinicalTrialSearchSpec,
)

API_BASE = "https://clinicaltrials.gov/api/v2/studies"


@dataclass(frozen=True)
class TrialRecord:
    nct_id: str
    title: str
    sponsor: str
    phase: str
    status: str
    conditions: str
    interventions: str
    start_date: str
    last_update: str
    source_query: str
    lane: str
    url: str


@dataclass(frozen=True)
class ClinicalTrialSignal:
    bucket: str
    title: str
    finding: str
    value: str
    evidence: str
    priority: int


@dataclass(frozen=True)
class ClinicalTrialsSummary:
    source_status: str
    fetched_at_utc: str
    total_trials: int
    active_trials: int
    lanes_covered: list[str]
    signals: list[ClinicalTrialSignal]
    trial_table: pd.DataFrame
    persistence_payload: list[dict[str, Any]]
    source_errors: list[str]

    @property
    def new_information(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "new_information"]

    @property
    def value_interpretation(self) -> list[str]:
        return [s.value for s in self.signals if s.bucket == "value"]

    @property
    def trend_inference(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "trend"]

    @property
    def positioning_implications(self) -> list[str]:
        return [s.finding for s in self.signals if s.bucket == "positioning"]


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(_extract_text(v) for v in value if _extract_text(v))
    if isinstance(value, dict):
        return ", ".join(_extract_text(v) for v in value.values() if _extract_text(v))
    return str(value).strip()


def _first_date(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("date") or value.get("startDate") or value.get("completionDate") or "")
    return _extract_text(value)


def _phase(protocol: dict[str, Any]) -> str:
    phases = protocol.get("designModule", {}).get("phases")
    text = _extract_text(phases)
    return text or "Not specified"


def _sponsor(protocol: dict[str, Any]) -> str:
    lead = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    return _extract_text(lead.get("name")) or "Unknown sponsor"


def _interventions(protocol: dict[str, Any]) -> str:
    arms = protocol.get("armsInterventionsModule", {}).get("interventions", []) or []
    names = []
    for item in arms:
        name = item.get("name") if isinstance(item, dict) else None
        if name:
            names.append(str(name))
    return ", ".join(dict.fromkeys(names)) or "Not specified"


def _conditions(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("conditionsModule", {}).get("conditions")) or "Not specified"


def _status(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("statusModule", {}).get("overallStatus")) or "Unknown"


def _title(protocol: dict[str, Any]) -> str:
    id_module = protocol.get("identificationModule", {})
    return _extract_text(id_module.get("briefTitle") or id_module.get("officialTitle")) or "Untitled trial"


def _nct_id(protocol: dict[str, Any]) -> str:
    return _extract_text(protocol.get("identificationModule", {}).get("nctId"))


def _record_from_study(study: dict[str, Any], spec: ClinicalTrialSearchSpec) -> TrialRecord | None:
    protocol = study.get("protocolSection", {}) if isinstance(study, dict) else {}
    nct_id = _nct_id(protocol)
    if not nct_id:
        return None
    status_module = protocol.get("statusModule", {})
    return TrialRecord(
        nct_id=nct_id,
        title=_title(protocol),
        sponsor=_sponsor(protocol),
        phase=_phase(protocol),
        status=_status(protocol),
        conditions=_conditions(protocol),
        interventions=_interventions(protocol),
        start_date=_first_date(status_module.get("startDateStruct")),
        last_update=_first_date(status_module.get("lastUpdatePostDateStruct")),
        source_query=spec.query,
        lane=spec.label,
        url=f"https://clinicaltrials.gov/study/{nct_id}",
    )


def _request_payload(params: dict[str, str]) -> dict[str, Any]:
    url = f"{API_BASE}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "NextCure-Intelligence-Prototype/0.9.13"})
    with urlopen(request, timeout=CLINICAL_TRIALS_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed public API endpoint
        return json.loads(response.read().decode("utf-8"))


def _fetch_spec(spec: ClinicalTrialSearchSpec) -> tuple[list[TrialRecord], str | None]:
    base_params = {
        "query.term": spec.query,
        "pageSize": str(CLINICAL_TRIALS_PAGE_SIZE),
        "format": "json",
    }
    attempts = [
        # Preferred if accepted by the upstream API: newest/most recently updated first.
        base_params | {"sort": "LastUpdatePostDate:desc"},
        # Safe fallback if the API rejects or changes sort syntax.
        base_params,
    ]
    last_error: str | None = None
    payload: dict[str, Any] | None = None
    for params in attempts:
        try:
            payload = _request_payload(params)
            break
        except Exception as exc:  # network/API failure should never break the dashboard
            last_error = f"{type(exc).__name__}: {exc}"

    if payload is None:
        return [], f"{spec.label}: {last_error or 'unknown upstream error'}"

    records: list[TrialRecord] = []
    for study in payload.get("studies", []) or []:
        record = _record_from_study(study, spec)
        if record is not None:
            records.append(record)
    return records, None


def _is_active(status: str) -> bool:
    text = status.lower()
    return any(token in text for token in ["recruiting", "active", "enrolling", "not yet recruiting"])


def _trial_table(records: list[TrialRecord]) -> pd.DataFrame:
    columns = [
        "Lane", "NCT ID", "Sponsor", "Phase", "Status", "Title",
        "Conditions", "Interventions", "Start Date", "Last Update", "URL",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([
        {
            "Lane": r.lane,
            "NCT ID": r.nct_id,
            "Sponsor": r.sponsor,
            "Phase": r.phase,
            "Status": r.status,
            "Title": r.title,
            "Conditions": r.conditions,
            "Interventions": r.interventions,
            "Start Date": r.start_date,
            "Last Update": r.last_update,
            "URL": r.url,
        }
        for r in records
    ])


def _summarize_lanes(records: list[TrialRecord]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for r in records:
        lane = summary.setdefault(r.lane, {"count": 0, "active": 0, "sponsors": set(), "phases": set()})
        lane["count"] += 1
        lane["active"] += 1 if _is_active(r.status) else 0
        lane["sponsors"].add(r.sponsor)
        lane["phases"].add(r.phase)
    for lane in summary.values():
        lane["sponsors"] = sorted(lane["sponsors"])
        lane["phases"] = sorted(lane["phases"])
    return summary



def _lane_records(records: list[TrialRecord], lane_name: str) -> list[TrialRecord]:
    return [r for r in records if r.lane == lane_name]


def _active_records(records: list[TrialRecord]) -> list[TrialRecord]:
    return [r for r in records if _is_active(r.status)]


def _sponsor_phrase(records: list[TrialRecord], limit: int = 4) -> str:
    sponsors = []
    for r in records:
        if r.sponsor and r.sponsor != "Unknown sponsor" and r.sponsor not in sponsors:
            sponsors.append(r.sponsor)
    if not sponsors:
        return "sponsor detail not clearly listed"
    if len(sponsors) <= limit:
        return ", ".join(sponsors)
    return ", ".join(sponsors[:limit]) + f", +{len(sponsors) - limit} more"


def _phase_phrase(phases: list[str] | set[str]) -> str:
    clean = [p for p in sorted(phases) if p and p != "Not specified"]
    return ", ".join(clean[:4]) if clean else "phase detail not consistently specified"


def _theme_hits(records: list[TrialRecord]) -> dict[str, int]:
    theme_terms = {
        "biomarker / patient-selection language": ["biomarker", "expression", "positive", "selected", "selection", "stratified", "molecular"],
        "combination strategy": ["combination", "combined", "plus", "with pembrolizumab", "with chemotherapy", "with paclitaxel"],
        "ovarian / gynecologic focus": ["ovarian", "fallopian", "peritoneal", "gynecologic", "gynaecologic"],
        "antibody / ADC modality language": ["adc", "antibody drug", "antibody-drug", "antibody", "conjugate"],
    }
    counts = {theme: 0 for theme in theme_terms}
    for r in records:
        haystack = " ".join([r.title, r.conditions, r.interventions]).lower()
        for theme, terms in theme_terms.items():
            if any(term in haystack for term in terms):
                counts[theme] += 1
    return {theme: count for theme, count in counts.items() if count > 0}


def _build_signals(records: list[TrialRecord], errors: list[str]) -> list[ClinicalTrialSignal]:
    signals: list[ClinicalTrialSignal] = []
    if not records:
        detail = "No usable ClinicalTrials.gov records were returned for the configured search lanes in this run."
        if errors:
            detail += " Source diagnostics were captured without interrupting the dashboard."
        return [
            ClinicalTrialSignal(
                bucket="new_information",
                title="ClinicalTrials.gov live pull unavailable",
                finding=detail,
                value="This is a source-health read, not an investment conclusion. It keeps the executive workflow stable when the upstream API is unavailable.",
                evidence="No normalized trial records available for this run.",
                priority=99,
            )
        ]

    lane_summary = _summarize_lanes(records)
    top_lanes = sorted(lane_summary.items(), key=lambda item: (item[1]["active"], item[1]["count"]), reverse=True)
    direct_lane_names = [name for name in ["CDH6 / Ovarian ADC", "B7-H4 ADC", "Ovarian ADC"] if name in lane_summary]
    side_lane_names = [name for name in ["Alzheimer's Side Channel", "Bone Disease Side Channel"] if name in lane_summary]
    direct_records = [r for r in records if r.lane in direct_lane_names]
    active_direct = _active_records(direct_records)
    themes = _theme_hits(records)
    direct_themes = _theme_hits(direct_records) if direct_records else {}

    if direct_lane_names:
        strongest_direct = sorted(
            ((lane, lane_summary[lane]) for lane in direct_lane_names),
            key=lambda item: (item[1]["active"], item[1]["count"]),
            reverse=True,
        )[0]
        lane_name, data = strongest_direct
        lane_recs = _lane_records(records, lane_name)
        signals.append(ClinicalTrialSignal(
            bucket="new_information",
            title="Direct trial landscape signal",
            finding=(
                f"Directly relevant trial activity is live in {', '.join(direct_lane_names)}. "
                f"The strongest direct lane in this run is {lane_name}, with {data['active']} active studies among {data['count']} returned records."
            ),
            value=(
                "This is useful because it confirms whether the target/indication universe is active in real clinical development, "
                "rather than relying only on stock-price movement or static peer lists."
            ),
            evidence=f"Representative sponsors surfaced: {_sponsor_phrase(lane_recs)}.",
            priority=1,
        ))
    else:
        leader, data = top_lanes[0]
        lane_recs = _lane_records(records, leader)
        signals.append(ClinicalTrialSignal(
            bucket="new_information",
            title="Live trial landscape signal",
            finding=(
                f"The live pull did not surface a direct CDH6/B7-H4/ovarian ADC concentration, "
                f"but {leader} showed the highest activity with {data['active']} active studies among {data['count']} returned records."
            ),
            value="This prevents the system from forcing a pipeline-specific conclusion when the live source does not support one.",
            evidence=f"Representative sponsors surfaced: {_sponsor_phrase(lane_recs)}.",
            priority=1,
        ))

    latest_active = sorted(_active_records(records), key=lambda r: r.last_update or "", reverse=True)[:2]
    if latest_active:
        signals.append(ClinicalTrialSignal(
            bucket="new_information",
            title="Recently updated active studies",
            finding="Recent active updates include " + "; ".join(
                f"{r.sponsor} in {r.lane} ({r.phase}, updated {r.last_update or 'N/A'})" for r in latest_active
            ) + ".",
            value="Recent active updates are more valuable than static trial existence because they show where clinical-development records are still moving.",
            evidence="; ".join(f"{r.nct_id}: {r.title}" for r in latest_active),
            priority=2,
        ))

    if direct_lane_names:
        phase_mix = sorted({p for lane in direct_lane_names for p in lane_summary[lane]["phases"]})
        sponsor_count = len({r.sponsor for r in direct_records if r.sponsor})
        signals.append(ClinicalTrialSignal(
            bucket="value",
            title="Why the direct-lane activity matters",
            finding=(
                f"The direct lanes contain {len(active_direct)} active/recruiting-style studies across {sponsor_count} sponsor(s), "
                f"with phase mix: {_phase_phrase(phase_mix)}."
            ),
            value=(
                "The valuable read is competitive density: active trials across direct lanes indicate the field is still being clinically developed, "
                "so leadership can separate true category inactivity from a visibility, differentiation, or timing problem."
            ),
            evidence="; ".join(f"{lane}: {lane_summary[lane]['active']} active / {lane_summary[lane]['count']} returned" for lane in direct_lane_names),
            priority=3,
        ))

    if direct_themes or themes:
        theme_source = direct_themes or themes
        top_theme, top_count = sorted(theme_source.items(), key=lambda item: item[1], reverse=True)[0]
        scope = "direct oncology" if direct_themes else "configured"
        signals.append(ClinicalTrialSignal(
            bucket="value",
            title="Repeated trial-design language",
            finding=f"The most repeated {scope} trial-language pattern was {top_theme}, appearing in {top_count} normalized record(s).",
            value=(
                "This is useful because repeated trial-design language shows what sponsors are emphasizing in active development, "
                "which is more actionable than a raw trial count."
            ),
            evidence=", ".join(f"{name}: {count}" for name, count in sorted(theme_source.items(), key=lambda item: item[1], reverse=True)[:3]),
            priority=4,
        ))

    if top_lanes:
        leader, data = top_lanes[0]
        second = top_lanes[1] if len(top_lanes) > 1 else None
        comparison = ""
        if second:
            comparison = f" The next densest lane was {second[0]} with {second[1]['active']} active studies."
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title="Trial-density hierarchy",
            finding=(
                f"Clinical activity is clustering most strongly in {leader}: {data['active']} active studies among {data['count']} returned records."
                + comparison
            ),
            value="Density hierarchy helps identify where development attention is concentrating before it is obvious from market performance alone.",
            evidence=f"{leader} phase mix: {_phase_phrase(data['phases'])}; sponsors: {', '.join(data['sponsors'][:4]) or 'N/A'}.",
            priority=5,
        ))

    ovarian_data = lane_summary.get("Ovarian ADC") or lane_summary.get("CDH6 / Ovarian ADC")
    if ovarian_data:
        ovarian_lane = "Ovarian ADC" if "Ovarian ADC" in lane_summary else "CDH6 / Ovarian ADC"
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title="Ovarian ADC lane read",
            finding=(
                f"Ovarian-linked ADC activity is not dormant in the live pull: {ovarian_data['active']} active studies were found "
                f"among {ovarian_data['count']} returned {ovarian_lane} records."
            ),
            value="This is the cleaner trend read: it tells leadership whether an ovarian ADC narrative is still clinically alive, independent of NXTC's current trading pattern.",
            evidence=f"Observed phases: {_phase_phrase(ovarian_data['phases'])}.",
            priority=6,
        ))

    for lane in side_lane_names[:2]:
        data = lane_summary[lane]
        signals.append(ClinicalTrialSignal(
            bucket="trend",
            title=f"{lane} side-channel read",
            finding=f"{lane} showed {data['active']} active studies among {data['count']} returned records in this run.",
            value="Side-channel activity is useful as optionality intelligence only; it should inform future exploration without distracting from the core oncology read.",
            evidence=f"Sponsors surfaced: {', '.join(data['sponsors'][:4]) or 'N/A'}.",
            priority=7,
        ))

    if direct_lane_names:
        direct_phrase = ", ".join(direct_lane_names)
        signals.append(ClinicalTrialSignal(
            bucket="positioning",
            title="Positioning implication",
            finding=(
                f"Because {direct_phrase} activity is present in live clinical records, NXTC's market performance should be judged against an active competitive landscape, "
                "not against a dormant category."
            ),
            value=(
                "If NXTC is lagging while direct lanes remain active, the executive question becomes whether the market needs clearer differentiation, better visibility, "
                "or catalyst confirmation."
            ),
            evidence="ClinicalTrials.gov records are normalized and available as supporting evidence below the Executive Summary.",
            priority=8,
        ))
    else:
        signals.append(ClinicalTrialSignal(
            bucket="positioning",
            title="Positioning implication",
            finding="The live clinical pull did not show a strong direct-lane concentration, so NXTC performance should be anchored more heavily to market/peer behavior in this run.",
            value="This keeps the system from overstating clinical-trial relevance when the live records do not support it.",
            evidence="ClinicalTrials.gov records are normalized and available as supporting evidence below the Executive Summary.",
            priority=8,
        ))

    return sorted(signals, key=lambda s: s.priority)

def build_clinical_trials_intelligence() -> ClinicalTrialsSummary:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    by_nct: dict[str, TrialRecord] = {}
    errors: list[str] = []

    for spec in CLINICAL_TRIAL_SEARCH_SPECS:
        records, error = _fetch_spec(spec)
        if error:
            errors.append(error)
        for record in records:
            existing = by_nct.get(record.nct_id)
            # Keep the highest-priority/source-specific lane for duplicates.
            if existing is None:
                by_nct[record.nct_id] = record
            else:
                existing_priority = next((s.priority for s in CLINICAL_TRIAL_SEARCH_SPECS if s.label == existing.lane), 99)
                if spec.priority < existing_priority:
                    by_nct[record.nct_id] = record

    records = list(by_nct.values())
    records.sort(key=lambda r: (r.last_update or "", r.nct_id), reverse=True)
    signals = _build_signals(records, errors)
    table = _trial_table(records)
    payload = [asdict(record) | {"fetched_at_utc": fetched_at, "source": "clinicaltrials.gov"} for record in records]
    active_count = sum(1 for r in records if _is_active(r.status))
    source_status = "live" if records else ("degraded" if errors else "empty")

    return ClinicalTrialsSummary(
        source_status=source_status,
        fetched_at_utc=fetched_at,
        total_trials=len(records),
        active_trials=active_count,
        lanes_covered=sorted({r.lane for r in records}),
        signals=signals,
        trial_table=table,
        persistence_payload=payload,
        source_errors=errors,
    )
