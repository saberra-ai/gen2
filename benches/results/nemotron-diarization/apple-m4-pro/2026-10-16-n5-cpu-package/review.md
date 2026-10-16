# Mac CPU packaging review

Spec and Standards independently reviewed `421d415...57490f5`.
Spec found no actionable issue in the claimed local relocation evidence and
independently checked all 92 loader-traced libraries, receipt hashes and tool tests.

Standards found that an entry symlink outside the SDK could bypass locked SDK
hash checks and produce a package without NeMo libraries. The fix now requires
the exact qualified native entry inside SDK/lib and checks its build-record hash
before discovery. All SDK build records are independently checked; a dependency
alias cannot redirect a locked SDK identity elsewhere.

The regression first failed without the new admission helper, then passed.
The malformed real CLI fixture was also rejected before publication. All 13 tool
tests pass. Repackaging the genuine SDK after the fix produced an identical
manifest and all 109 checksummed files matching the package used by the eight
executed native tests. No runtime binary changed in this admission correction.

Broader N5/quality gates remain open as listed in the receipt.
