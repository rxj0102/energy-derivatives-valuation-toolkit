import pytest

from energy_toolkit.power_price_model import GasPriceParams, PowerMarketSimulator, PowerPriceParams


@pytest.fixture(scope="session")
def small_scenario():
    """A small, fast joint gas/power scenario shared across tests that just
    need *some* realistic-looking simulated data (not testing the simulator
    itself)."""
    sim = PowerMarketSimulator()
    return sim.simulate_paths(n_years=1.0, n_paths=120, seed=2024)


@pytest.fixture(scope="session")
def medium_scenario():
    """A slightly larger scenario for tests needing more statistical power
    (e.g. seasonal / jump-frequency checks) while staying fast."""
    sim = PowerMarketSimulator()
    return sim.simulate_paths(n_years=3.0, n_paths=300, seed=99)


@pytest.fixture(scope="session")
def default_gas_params():
    return GasPriceParams()


@pytest.fixture(scope="session")
def default_power_params():
    return PowerPriceParams()
