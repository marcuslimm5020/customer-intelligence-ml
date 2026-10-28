# Data and validation design (initial)

This is the first modeling design for the UCI Online Retail data. It is intentionally simple and should be revisited after exploratory analysis.

## Transaction cleaning rules

1. Remove exact duplicate transaction rows.
2. Exclude rows without `CustomerID`, since they cannot be used for customer-level features or labels.
3. Exclude invoices whose `InvoiceNo` starts with `C` (cancellations/credit notes).
4. Exclude rows with `Quantity <= 0` or `UnitPrice <= 0`. This models positive sales activity; returns and non-sale adjustments do not count as purchases or contribute to these initial purchase features.
5. Treat each remaining row as a sale line. A customer is considered to have purchased in a horizon if they have at least one remaining sale line in it.

Applying those rules leaves 392,692 rows and 4,338 customers, after deduplicating the 541,909 source rows. The source contains 5,268 exact duplicates, 135,080 rows with missing customer IDs, 9,288 cancellation rows, 10,624 non-positive-quantity rows and 2,517 non-positive-price rows. These categories overlap, so their counts should not be added together. A quick check found no regular invoice IDs matching cancellation IDs after removing the leading `C`; cancellation pairing therefore needs no additional original-invoice removal under this dataset's IDs.

## Prediction example

Use monthly snapshots on the first of each month. For a cutoff date `t`:

- Features use the 90 calendar days in `[t - 90 days, t)`.
- The eligible population is customers with at least one cleaned sale line in that observation window.
- The label is 1 if an eligible customer has at least one cleaned sale line in `[t, t + 30 days)`, otherwise 0.
- No information dated on or after `t` may enter that row's features.

This defines a practical use case: rank customers who purchased recently by their chance of purchasing again in the next 30 days. It does not score dormant customers with no purchase in the prior 90 days.

## Chronological split

The dataset runs from 2010-12-01 through 2011-12-09, so these complete monthly examples fit:

- Training cutoffs: 2011-03-01 through 2011-08-01 (six monthly snapshots)
- Validation cutoffs: 2011-09-01 and 2011-10-01
- Final test cutoff: 2011-11-01

The November outcome window ends 2011-12-01, within the available data. Do not use December as a test snapshot because its 30-day outcome is incomplete. The current counts of eligible customers and positive labels per cutoff are in the README.

## Limits to keep in view

The data covers about one year, the final test is one month, and consecutive monthly examples share customers and overlapping observation windows. This is a temporal holdout for a first portfolio project, not evidence of performance across years or retailers. Report validation and final test results separately and avoid random row splits. The 90/30-day windows, active-customer scope, and treatment of returns should be checked against the intended marketing use case before treating the results as business-ready.
