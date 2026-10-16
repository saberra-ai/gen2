# N4 development review

The Standards and Spec axes independently reviewed `6b2e0ba...c8f3690` and
`a47f209...11103a5`. Standards reported no actionable findings. Spec found two
implemented-scope defects, both fixed and rechecked:

- `a47f209`: overlap now sums activity common to every reported speaker;
  two public regressions reproduced the earlier failures and pass after the fix.
- `b65d2ad`: listing history retains the running session's authoritative snapshot;
  true disk recovery marks its lost native cache interrupted. The regression
  reproduced the false-interruption failure and now passes both cases.

Persisted parity initially compared floating-point boundaries exactly. The
captured diagnostic has identical 35 turns/IDs and at most 3.55e-15 seconds boundary
rounding after JSON persistence. The corrected check uses the established
1e-6-second native-reference tolerance and exact IDs/counts; a regression rejects
1 ms boundary changes and speaker changes. The 80.04-second concurrent live run
then passed. In-memory capture-path comparison remains exact.

Final targeted Spec recheck passed all three live tests and found no remaining
defect in the corrections. These reviews cover implemented behavior; they do
not close corpus quality, physical microphone/native visual or N5 release gates.
