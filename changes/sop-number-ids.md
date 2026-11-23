### Fixed - SOP revisions could add a number that appeared only inside a record id

- The revise check ("no new numbers") counted the digits inside record ids such as `run:2f30d1c7...` as numbers the draft already held, so a revision adding "30" was accepted when a run id happened to contain "30". Digits inside long hex ids no longer count. This also fixes the intermittent `test_revise_refuses_a_number_change` failure. No ML outputs change.
