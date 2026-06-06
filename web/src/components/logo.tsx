type LogoProps = {
  className?: string
}

/**
 * HTTYR brand mark — a minimal outline robot head. Drawn with currentColor
 * strokes so it inherits the surrounding text color and stays crisp at any size.
 */
export function Logo({ className = 'h-7 w-7' }: LogoProps) {
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="HTTYR — How to train your robot"
      shapeRendering="geometricPrecision"
    >
      <path d="M16 5.6V3" />
      <circle cx="16" cy="2.6" r="1.25" fill="currentColor" stroke="none" />
      <rect x="5.6" y="5.6" width="20.8" height="20.8" rx="7" />
      <path d="M2.9 14.4v3.6" />
      <path d="M29.1 14.4v3.6" />
      <rect x="9.4" y="11.6" width="13.2" height="9" rx="4.5" />
      <circle cx="13" cy="16.1" r="1.45" fill="currentColor" stroke="none" />
      <circle cx="19" cy="16.1" r="1.45" fill="currentColor" stroke="none" />
    </svg>
  )
}

type BrandProps = {
  logoClassName?: string
  nameClassName?: string
  showSubtitle?: boolean
}

/**
 * Full lockup: outline mark + short-form wordmark "HTTYR" with the expanded
 * name "How to train your robot" set quietly underneath.
 */
export function Brand({
  logoClassName = 'h-7 w-7',
  nameClassName = 'text-[15px]',
  showSubtitle = true,
}: BrandProps) {
  return (
    <span className="flex items-center gap-2.5">
      <span className="text-white">
        <Logo className={logoClassName} />
      </span>
      <span className="flex flex-col leading-none">
        <span className={`brand-wordmark tracking-[0.18em] ${nameClassName}`}>HTTYR</span>
        {showSubtitle && (
          <span
            className="mt-1 text-[11px] italic leading-none text-[var(--foreground-tertiary)]"
            style={{ fontFamily: 'Georgia, serif' }}
          >
            How to train your robot
          </span>
        )}
      </span>
    </span>
  )
}

export default Logo
