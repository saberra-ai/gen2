export function Icon({ name }: { name: string }) {
  return (
    <svg aria-hidden="true">
      <use href={`#i-${name}`} />
    </svg>
  );
}
export function IconLibrary() {
  return (
    <svg
      className="icon-library"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <defs>
        <symbol id="i-wave" viewBox="0 0 24 24">
          <path d="M4 10v4m4-8v12m4-15v18m4-15v12m4-8v4" />
        </symbol>
        <symbol id="i-mic" viewBox="0 0 24 24">
          <rect x="9" y="3" width="6" height="12" rx="3" />
          <path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8" />
        </symbol>
        <symbol id="i-upload" viewBox="0 0 24 24">
          <path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" />
        </symbol>
        <symbol id="i-arrow" viewBox="0 0 24 24">
          <path d="M4 12h15m-6-6 6 6-6 6" />
        </symbol>
        <symbol id="i-plus" viewBox="0 0 24 24">
          <path d="M12 5v14M5 12h14" />
        </symbol>
        <symbol id="i-copy" viewBox="0 0 24 24">
          <rect x="8" y="8" width="12" height="13" rx="2" />
          <path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" />
        </symbol>
        <symbol id="i-download" viewBox="0 0 24 24">
          <path d="M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4" />
        </symbol>
        <symbol id="i-file" viewBox="0 0 24 24">
          <path d="M14 3H5v18h14V8l-5-5v5h5M8 12h8m-8 4h6" />
        </symbol>
        <symbol id="i-x" viewBox="0 0 24 24">
          <path d="m6 6 12 12M6 18 18 6" />
        </symbol>
        <symbol id="i-lock" viewBox="0 0 24 24">
          <rect x="5" y="10" width="14" height="11" rx="2" />
          <path d="M8 10V7a4 4 0 0 1 8 0v3m-4 4v3" />
        </symbol>
      </defs>
    </svg>
  );
}
