"""
Test suite for the quickstart tutorial.

This module tests that all code examples in docs/source/quickstart.rst
are executable and produce expected results.
"""

import numpy as np
import pytest
import xarray as xr


class TestQuickstartTutorial:
    """Test all code examples from the quickstart tutorial."""

    @pytest.fixture
    def sample_spi_dataset(self):
        """Create sample SPI dataset as shown in quickstart."""
        # Create a sample dataset
        nx, ny, nt = 20, 20, 365  # 20x20 grid, 365 days

        # Create coordinates
        lats = np.linspace(-90, 90, nx)
        lons = np.linspace(-180, 180, ny)
        times = xr.date_range("1950-01-01", periods=nt, freq="D")

        # Create standard normal data for SPI values
        np.random.seed(42)  # For reproducibility
        spi_data = np.random.normal(0, 1, size=(nx, ny, nt))

        # Create xarray Dataset
        spi = xr.Dataset(
            data_vars={"SPI1": (["lat", "lon", "time"], spi_data)},
            coords={"lat": lats, "lon": lons, "time": times},
        )

        return spi

    def test_imports(self):
        """Test that all required imports work."""
        import numpy as np
        import xarray as xr
        import dominosee as ds

        assert np is not None
        assert xr is not None
        assert ds is not None

    def test_create_sample_data(self, sample_spi_dataset):
        """Test creating sample SPI dataset."""
        spi = sample_spi_dataset

        # Verify dataset structure
        assert isinstance(spi, xr.Dataset)
        assert "SPI1" in spi.data_vars
        assert set(spi.dims) == {"lat", "lon", "time"}
        assert spi.dims["lat"] == 20
        assert spi.dims["lon"] == 20
        assert spi.dims["time"] == 365

        # Verify coordinates
        assert "lat" in spi.coords
        assert "lon" in spi.coords
        assert "time" in spi.coords

    def test_extract_extreme_events(self, sample_spi_dataset):
        """Test extracting extreme events using get_event."""
        from dominosee.eventorize import get_event

        spi = sample_spi_dataset

        # Extract drought events (SPI < -1.0)
        da_event = get_event(
            spi.SPI1, threshold=-1.0, extreme="below", event_name="drought"
        )

        # Verify event extraction
        assert isinstance(da_event, xr.DataArray)
        assert da_event.dtype == bool
        assert da_event.shape == spi.SPI1.shape
        assert da_event.name == "drought"

        # Verify attributes
        assert da_event.attrs["threshold"] == -1.0
        assert da_event.attrs["extreme"] == "below"
        assert da_event.attrs["event_name"] == "drought"

        # Verify some events were detected
        assert da_event.sum() > 0

    def test_eca_precursor_trigger(self, sample_spi_dataset):
        """Test ECA precursor and trigger calculation."""
        from dominosee.eventorize import get_event
        from dominosee.eca import (
            get_eca_precursor_from_events,
            get_eca_trigger_from_events,
        )

        spi = sample_spi_dataset
        da_event = get_event(
            spi.SPI1, threshold=-1.0, extreme="below", event_name="drought"
        )

        # Calculate precursor and trigger events
        da_precursor = get_eca_precursor_from_events(
            eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
        )

        da_trigger = get_eca_trigger_from_events(
            eventA=da_event, eventB=da_event, delt=10, sym=True, tau=0
        )

        # Verify precursor results
        assert isinstance(da_precursor, xr.DataArray)
        assert set(da_precursor.dims) == {"latA", "lonA", "latB", "lonB"}
        assert da_precursor.sizes["latA"] == 20
        assert da_precursor.sizes["lonA"] == 20
        assert da_precursor.sizes["latB"] == 20
        assert da_precursor.sizes["lonB"] == 20
        assert "eca_params" in da_precursor.attrs

        # Verify trigger results
        assert isinstance(da_trigger, xr.DataArray)
        assert set(da_trigger.dims) == {"latA", "lonA", "latB", "lonB"}
        assert da_trigger.sizes["latA"] == 20
        assert da_trigger.sizes["lonA"] == 20
        assert da_trigger.sizes["latB"] == 20
        assert da_trigger.sizes["lonB"] == 20
        assert "eca_params" in da_trigger.attrs

    def test_eca_confidence(self, sample_spi_dataset):
        """Test ECA confidence calculation."""
        from dominosee.eventorize import get_event
        from dominosee.eca import (
            get_eca_precursor_from_events,
            get_eca_trigger_from_events,
            get_eca_precursor_confidence,
            get_eca_trigger_confidence,
        )

        spi = sample_spi_dataset
        da_event = get_event(
            spi.SPI1, threshold=-1.0, extreme="below", event_name="drought"
        )

        da_precursor = get_eca_precursor_from_events(
            eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
        )

        da_trigger = get_eca_trigger_from_events(
            eventA=da_event, eventB=da_event, delt=10, sym=True, tau=0
        )

        # Calculate statistical confidence
        da_prec_conf = get_eca_precursor_confidence(
            precursor=da_precursor, eventA=da_event, eventB=da_event
        )

        da_trig_conf = get_eca_trigger_confidence(
            trigger=da_trigger, eventA=da_event, eventB=da_event
        )

        # Verify precursor confidence
        assert isinstance(da_prec_conf, xr.DataArray)
        assert set(da_prec_conf.dims) == {"latA", "lonA", "latB", "lonB"}
        assert da_prec_conf.min() >= 0.0
        assert da_prec_conf.max() <= 1.0

        # Verify trigger confidence
        assert isinstance(da_trig_conf, xr.DataArray)
        assert set(da_trig_conf.dims) == {"latA", "lonA", "latB", "lonB"}
        assert da_trig_conf.min() >= 0.0
        assert da_trig_conf.max() <= 1.0

    def test_construct_network(self, sample_spi_dataset):
        """Test network construction from ECA confidence levels."""
        from dominosee.eventorize import get_event
        from dominosee.eca import (
            get_eca_precursor_from_events,
            get_eca_trigger_from_events,
            get_eca_precursor_confidence,
            get_eca_trigger_confidence,
        )
        from dominosee.network import get_link_from_confidence

        spi = sample_spi_dataset
        da_event = get_event(
            spi.SPI1, threshold=-1.0, extreme="below", event_name="drought"
        )

        da_precursor = get_eca_precursor_from_events(
            eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
        )

        da_trigger = get_eca_trigger_from_events(
            eventA=da_event, eventB=da_event, delt=10, sym=True, tau=0
        )

        da_prec_conf = get_eca_precursor_confidence(
            precursor=da_precursor, eventA=da_event, eventB=da_event
        )

        da_trig_conf = get_eca_trigger_confidence(
            trigger=da_trigger, eventA=da_event, eventB=da_event
        )

        # Create network from ECA confidence levels
        da_link = get_link_from_confidence(
            da_prec_conf, 0.99
        ) & get_link_from_confidence(da_trig_conf, 0.99)

        # Verify network structure
        assert isinstance(da_link, xr.DataArray)
        assert da_link.dtype == bool
        assert set(da_link.dims) == {"latA", "lonA", "latB", "lonB"}

        # Calculate network density
        density = da_link.sum().values / da_link.size * 100

        # Verify density is reasonable
        assert 0 <= density <= 100
        assert isinstance(density, (int, float, np.number))

    def test_complete_workflow(self, sample_spi_dataset):
        """Test the complete quickstart workflow end-to-end."""
        import numpy as np
        import xarray as xr
        from dominosee.eventorize import get_event
        from dominosee.eca import (
            get_eca_precursor_from_events,
            get_eca_trigger_from_events,
            get_eca_precursor_confidence,
            get_eca_trigger_confidence,
        )
        from dominosee.network import get_link_from_confidence

        # Use the sample dataset
        spi = sample_spi_dataset

        # Extract drought events (SPI < -1.0)
        da_event = get_event(
            spi.SPI1, threshold=-1.0, extreme="below", event_name="drought"
        )

        # Calculate precursor and trigger events
        da_precursor = get_eca_precursor_from_events(
            eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
        )

        da_trigger = get_eca_trigger_from_events(
            eventA=da_event, eventB=da_event, delt=10, sym=True, tau=0
        )

        # Calculate statistical confidence
        da_prec_conf = get_eca_precursor_confidence(
            precursor=da_precursor, eventA=da_event, eventB=da_event
        )

        da_trig_conf = get_eca_trigger_confidence(
            trigger=da_trigger, eventA=da_event, eventB=da_event
        )

        # Create network from ECA confidence levels
        da_link = get_link_from_confidence(
            da_prec_conf, 0.99
        ) & get_link_from_confidence(da_trig_conf, 0.99)

        # Calculate network density
        density = da_link.sum().values / da_link.size * 100

        # Verify complete workflow succeeded
        assert isinstance(da_event, xr.DataArray)
        assert isinstance(da_precursor, xr.DataArray)
        assert isinstance(da_trigger, xr.DataArray)
        assert isinstance(da_prec_conf, xr.DataArray)
        assert isinstance(da_trig_conf, xr.DataArray)
        assert isinstance(da_link, xr.DataArray)
        assert 0 <= density <= 100

        # Print network density as in the tutorial
        print(f"Network density: {density:.2f}%")
