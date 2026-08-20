from typing import Optional

from ..type_aliases import DataArray

class CeilometerRangeCorrectionTransformer:
    def transform(self, beta: DataArray, range_var: DataArray) -> DataArray:
        """Apply range correction to ceilometer backscatter data.

        Corrects for geometric beam spreading with distance.
        Raw signal ∝ 1/r², so multiply by r² to get range-corrected signal (RCS).

        Args:
            beta (DataArray): Raw ceilometer backscatter with height coordinate
            range_var (DataArray): Range coordinate variable (meters)

        Returns:
            DataArray: Range-corrected backscatter signal
        """
        if beta.attrs.get("range_corrected", False):
            raise ValueError("Data already range-corrected")

        # r² correction (convert meters to km for numerical stability)
        range_correction_factor = (range_var * 1e-3) ** 2
        beta = beta * range_correction_factor
        beta.attrs["range_corrected"] = 1

        return beta

    def inverse_transform(self, beta: DataArray, range_var: DataArray, max_range: Optional[float] = None) -> DataArray:
        """Apply inverse range correction to ceilometer backscatter data.

        Applies inverse of r² range correction following CloudnetPy processing
        scheme to convert range-corrected signal (RCS) to raw backscatter
        signal.

        Args:
            beta (DataArray): Range-corrected signal (RCS) with height coordinate
            range_var (DataArray): Range coordinate variable
            max_range (float | None): Maximum range to apply inverse correction

        Returns:
            DataArray: Raw ceilometer backscatter signal
        """

        if not beta.attrs.get("range_corrected", False):
            raise ValueError("Input data must be range-corrected before applying inverse transform.")

        # Multiply by range squared to correct for geometric beam spreading
        range_correction_factor = (range_var * 1e-3) ** 2
        if max_range:
            range_correction_factor = range_correction_factor.where(range_var <= max_range, 1.0)

        # Apply correction while preserving attributes and coordinates
        beta[:] = beta[:] / range_correction_factor[:]

        # Update attributes to indicate range correction has been applied
        beta.attrs["range_corrected"] = 0

        return beta