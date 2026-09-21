"""
Human workload accounting: review arrivals, queue utilization, and operator fatigue modeling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadOutcomes:
    alert_rate_per_hr: float
    total_review_hours: float
    queue_utilization_rho: float
    is_overloaded: bool
    avoided_hours: float = 0.0


class WorkloadAccountingEngine:
    def __init__(
        self,
        service_capacity_mu_per_hr: float = 60.0,
        review_duration_seconds: float = 30.0,
        sampling_rate_fps: float = 30.0,
        functional_unit_units: int = 1000,
    ):
        self.mu = service_capacity_mu_per_hr
        self.t_review = review_duration_seconds
        self.fps = sampling_rate_fps
        self.n_units = functional_unit_units

    def evaluate(
        self, alert_rate_per_hr: float, baseline_hours: float = None
    ) -> WorkloadOutcomes:
        # Throughput per hour = FPS * 3600
        throughput_hr = self.fps * 3600.0
        # Hours taken to inspect functional unit
        inspection_time_hr = self.n_units / max(1.0, throughput_hr)

        # Expected review alerts arriving per 1000 units
        n_alerts = alert_rate_per_hr * inspection_time_hr

        # Total review hours = n_alerts * (t_review / 3600.0)
        review_hours = n_alerts * (self.t_review / 3600.0)

        # Queue utilization rho = lambda / mu
        rho = alert_rate_per_hr / max(1e-6, self.mu)
        is_overloaded = bool(rho >= 1.0)

        avoided = 0.0
        if baseline_hours is not None:
            avoided = baseline_hours - review_hours

        return WorkloadOutcomes(
            alert_rate_per_hr=alert_rate_per_hr,
            total_review_hours=review_hours,
            queue_utilization_rho=rho,
            is_overloaded=is_overloaded,
            avoided_hours=avoided,
        )