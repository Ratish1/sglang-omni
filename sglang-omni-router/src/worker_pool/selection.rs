use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::config::RoutingStrategy;

/// Generation policy state. Least-request observation, ordering, and exact
/// reservation occur while the short unit-valued guard is held.
pub(super) struct Selector {
    strategy: RoutingStrategy,
    cursor: AtomicU64,
    least_requests: Mutex<()>,
}

impl Selector {
    pub(super) fn new(strategy: RoutingStrategy) -> Self {
        Self {
            strategy,
            cursor: AtomicU64::new(0),
            least_requests: Mutex::new(()),
        }
    }

    pub(super) fn least_requests_guard(
        &self,
        candidate_count: usize,
    ) -> Option<MutexGuard<'_, ()>> {
        if self.strategy != RoutingStrategy::LeastRequests || candidate_count < 2 {
            return None;
        }
        Some(
            self.least_requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    pub(super) const fn strategy(&self) -> RoutingStrategy {
        self.strategy
    }

    pub(super) fn start(&self, pool_size: usize) -> usize {
        if pool_size == 0 {
            return 0;
        }
        let sequence = self.cursor.fetch_add(1, Ordering::Relaxed);
        u64::try_from(pool_size)
            .ok()
            .and_then(|size| usize::try_from(sequence % size).ok())
            .unwrap_or(0)
    }
}
