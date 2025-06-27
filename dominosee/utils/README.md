# Dominosee Utilities

This directory contains utility functions that support the main functionality of the Dominosee package.

## Available Utilities

### Blocking (`blocking.py`)

The blocking module provides functionality for processing large datasets in smaller, memory-efficient blocks.
This is particularly useful for Event Coincidence Analysis (ECA) with very large spatial dimensions.

#### Key Functions

- **process_blocks**: Generic function to apply any function to blocks of data
- **combine_blocks**: Combines saved block files into a single dataset, with optional region selection
- **process_eca_blockwise**: Specialized wrapper for ECA functions, handles dimension renaming and common parameters

#### Example Usage

```python
from dominosee.eca import get_eca_precursor_from_events
from dominosee.utils.blocking import process_eca_blockwise, combine_blocks
import os

# Process data in blocks and save to disk
process_eca_blockwise(
    get_eca_precursor_from_events,
    event_a,
    event_b,
    output_dir="eca_results",
    block_size=1000,
    delt=2,
    sym=True,
    tau=0
)

# Later, load only a specific region
region = {"locationA": slice(0, 10), "locationB": slice(0, 10)}
result = combine_blocks(
    os.path.join("eca_results", "eca_block_*.nc"),
    region=region
)
```

See the `examples/eca_blockwise_example.py` file for a complete working example.

## Memory Considerations

For large datasets, the main limitation is often memory rather than computation time.
Here are approximate memory requirements for different block sizes:

| Block Size | Approximate Memory |
|------------|-------------------|
| 1,000      | ~4 GB             |
| 5,000      | ~100 GB           |
| 10,000     | ~400 GB           |

Choose a block size that fits comfortably in your available memory, with room for overhead. 