### Fixed - uploaded sheets: no invented values, ids and labels as written, rows always stored

- A cell such as `3.25` followed by a non-breaking space counted as a number but was converted to missing, so in a blank-is-absent column (a concentration) it was fitted as 0.0 without being reported, and in other columns the fit failed. One reader (`kalos.portal.cells.parse_number` / `to_numbers`) now decides both; whitespace is trimmed, and `1_000`, non-ASCII digits and `inf` are reported as non-numeric.
- A sheet with a date column saved no rows (JSON encoding failed) while the upload returned 200. Dates and times are stored as ISO text, and every upload response carries `rows_stored`.
- Run ids and codes written with leading zeros (`001`) keep them, in CSV and Excel text cells; `001` and `0001` no longer collide.
- Numeric category codes with a blank in the column were shown as `1.0`, `2.0`; they are now `1`, `2` everywhere (levels, proposals, effects).
- Existing fits on sheets without these cells are unchanged (full suite and goldens identical).
