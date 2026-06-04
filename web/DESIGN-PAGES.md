# DataMarket Page Design Plan

This is a design plan only. No implementation details or code are included here.

Note on process: `/teach-impeccable` and `/critique` were not available as local tools in this workspace, so the critique below is based on direct review of the current web pages and layouts in `web/src/app`.

## Current State Critique

The current product is clean and usable, but visually generic. It relies on gray cards, black buttons, and small text with little differentiation between information tiers. The layouts are structurally sound, yet they do not express enough trust, precision, or marketplace value for either Labs or Collectors.

Specific issues to correct in the next pass:

- The pages are too form-linear. The most important information often reads as one uninterrupted stack rather than a guided workflow.
- Data density is uneven. Key financial numbers, task incentives, and progress signals should feel immediate, but today they blend into the same treatment as supporting metadata.
- Color is functional but underused. Role distinction exists, yet the app does not use color to establish mood, scannability, or semantic emphasis.
- Task requirement tags and status states are present, but they do not carry enough visual meaning to aid fast decision-making.
- The current UI feels like a starter admin panel. It needs a sharper visual system that feels operational, reliable, and payout-oriented.

## Shared Design System Direction

### Product posture

DataMarket should feel like an operational marketplace, not a marketing site and not a consumer social app. The UI should communicate:

- Labs are specifying tasks with rigor and intent.
- Collectors are evaluating task worth quickly.
- Earnings and completion status are highly legible.

### Core visual direction

- Base canvas: soft warm-white or cool paper-gray background, not pure white.
- Surfaces: flat or lightly elevated panels with crisp borders, 8px radius maximum.
- Accent split by role:
  - Lab: deep cobalt or indigo as the dominant action color.
  - Collector: rich green as the dominant action color.
- Neutrals should stay restrained and slightly warm to prevent a sterile dashboard look.
- Use color semantically, not decoratively. Tags, payouts, deadlines, and statuses should each have a consistent meaning.

### Typography

- Keep the existing Geist family for now to stay close to the current app stack, but use it with more discipline.
- Page titles should be larger and denser, with stronger weight and tighter spacing.
- Section titles should step down clearly from page titles.
- Metadata should be smaller but still high-contrast enough for repeated scanning.
- Numeric values should be visually promoted, especially bounty, total earnings, submission counts, and dates.
- Monospace should be reserved for IDs, payout references, and machine-like values only.

### Shared component language

- Use full-width page bands and framed tools, not floating marketing cards.
- Use segmented role markers, colored tags, stat tiles, progress meters, upload wells, timeline rows, and chart panels.
- Inputs should feel production-grade: stable height, clear labels, muted helper text, visible focus, and strong grouping.
- Every page should have one dominant primary action and one clearly secondary cluster of supporting information.

### Impeccable principles to apply

- Clarity before decoration: every color and shape must help a user decide faster.
- One dominant action per screen: primary tasks should never compete visually.
- Stable information architecture: key metrics, payout values, and task state should always occupy predictable zones.
- Dense but calm: more useful information can be shown, but spacing and grouping must prevent noise.
- Status should be self-evident: approved, pending, full, deadline, and requirement types should be understood at a glance.

## Page 1: Lab — Create Task

### Goal

Help a Lab define a collection task with confidence, while making the form feel like a structured brief rather than a generic admin form.

### Layout and visual hierarchy

- Use a two-column desktop layout.
- Left column: the main task definition form.
- Right column: a sticky summary rail showing payout math, task readiness, and a live preview of visible collector-facing details.
- On mobile, collapse to one column, with the summary rail moving beneath the first section and again above the final submit action.

Recommended content order:

1. Page header with title, short instruction, and role cue.
2. Task brief section:
   - title
   - description
3. Reference assets section:
   - upload area for image/video references
4. Requirements section:
   - colored tags for capture constraints
5. Payout and volume section:
   - bounty per submission
   - submissions required
6. Timing section:
   - deadline picker
7. Sticky summary:
   - estimated total spend
   - visible requirement tags
   - completion checklist
8. Primary submit action

### Component choices

- Header band with role badge and short explanatory copy.
- Framed form sections with section titles and helper text.
- Large drag-and-drop upload well with thumbnail strip below.
- Selectable requirement chips, not checkboxes, for tags like `Outdoor`, `Indoor`, `Motion`, `Monochrome`.
- Currency input with fixed currency prefix.
- Stepper or numeric field for submission count.
- Date-time picker in a framed control.
- Sticky summary panel with:
  - total projected spend
  - task completeness checklist
  - collector preview card

### Color usage

- Lab blue should drive the primary CTA, active tags, focus states, and summary emphasis.
- Requirement tags should each have a distinct but muted semantic color family:
  - `Outdoor`: moss or green
  - `Indoor`: slate or steel blue
  - `Motion`: amber or orange
  - `Monochrome`: graphite
- Use a subtle warning tone for missing required fields or incomplete sections.
- Keep the main form backgrounds neutral so the tags and payout fields carry the visual energy.

### Typography decisions

- Page title should read like a work surface heading, not a marketing headline.
- Section headings should be compact and operational.
- Helper text should be concise and placed directly under labels or section titles.
- Payout values in the summary should be the strongest numbers on the page.
- Tag labels should be uppercase or small caps only if they remain highly legible.

### Impeccable principles that apply

- The form should feel composed, not endless.
- The sticky summary should reduce uncertainty and prevent form fatigue.
- The collector preview should keep Labs aware of what they are actually publishing.
- Uploads should feel like evidence attachment, not a decorative media step.

## Page 2: Collector — Task View

### Goal

Help a Collector decide quickly whether a task is worth doing, understand exactly what is required, and clearly see whether submission must happen in the iOS app or has already been completed there.

### Layout and visual hierarchy

- Use a split hierarchy instead of one stacked card.
- Top band:
  - task title
  - bounty
  - spots left
  - deadline
  - submission state
- Main body on desktop:
  - left: task description, references, requirements
  - right: action panel for iOS handoff or submitted confirmation
- On mobile:
  - top metrics first
  - action panel immediately after
  - details and references below

Recommended content order:

1. Back link
2. Hero task header with payout prominence
3. Action panel:
   - `Open in iPhone app` if not submitted
   - confirmation state if submitted
4. Task description
5. Reference media gallery
6. Requirement tags and capture checklist
7. Secondary metadata

### Component choices

- Header block with dominant payout number and compact metadata row.
- Status pill for `Open`, `Submitted`, `Under review`, `Approved`, `Rejected`, `Full`.
- Reference gallery with one primary preview and smaller supporting thumbnails or video stills.
- Requirement chip row with semantic colors matching the Create Task page.
- Collector action panel containing:
  - primary app-launch button
  - task ID block with copy affordance
  - explanatory text about iOS-only submission
- Submitted confirmation card with date, status icon, and payout outcome if approved.

### Color usage

- Collector green should own the primary action and positive payout states.
- Submitted and approved states should use calm greens, not loud success neon.
- Pending review should use muted amber.
- Rejected should use restrained red, mostly for icon and label, not full-surface fill.
- Reference media should add visual richness; the rest of the page should stay controlled so media is easy to inspect.

### Typography decisions

- Task title should be bold and direct.
- Bounty should be the largest numeric element on the page.
- Metadata should sit in a tight top row with strong contrast.
- Description should use slightly larger body text than today for readability.
- Confirmation text should be concise and confidence-building, not overly verbose.

### Impeccable principles that apply

- The payout and action state must be obvious within the first viewport.
- The iOS-only limitation should be explicit, but it should not dominate the page.
- The submitted state should feel reassuring and final enough that a collector does not wonder what to do next.
- Reference material should be treated as evidence and instruction, not generic media.

## Page 3: Collector — Earnings

### Goal

Make earnings feel tangible, trustworthy, and trackable. This page should answer three questions immediately: how much have I earned, how is it changing over time, and where did each earning come from.

### Layout and visual hierarchy

- Use a top-heavy financial layout.
- First viewport should show:
  - total earnings
  - pending earnings
  - earnings-over-time graph
- Below that, show the transaction log with date, task, amount, and state.

Recommended content order:

1. Page header with title and small explanatory line
2. Financial summary row:
   - total earned
   - pending
   - optionally paid out if data exists later
3. Chart panel:
   - line or bar chart for earnings over time
4. Log section:
   - earnings rows sorted newest first
5. Empty-state guidance if no earnings exist

### Component choices

- Large total earnings tile with the strongest numeric emphasis in the app.
- Secondary tiles for pending and optional future payout metrics.
- Framed chart panel with clear date range selector if data density justifies it later.
- Earnings log as a table-like list with predictable columns:
  - task or source
  - date
  - amount
  - status
- Compact status badges for `pending` and `approved`.
- Empty state that still preserves the structure of the financial page rather than dropping into a whimsical illustration.

### Color usage

- Collector green should anchor the page, especially the total earnings figure and positive chart stroke.
- Neutral gridlines and chart scaffolding should stay quiet.
- Pending amounts should use amber to distinguish non-withdrawable funds.
- Row hover or focus states should be subtle and operational.
- Avoid rainbow financial charts. One primary series color is enough.

### Typography decisions

- Total earnings should be oversized and visually dominant.
- Supporting financial labels should be compact, uppercase only if contrast remains strong.
- Chart labels should be small and unobtrusive.
- Log rows should prioritize amount and task title first, then date and status.
- Currency formatting should be consistent and visually aligned so amounts compare easily.

### Impeccable principles that apply

- The first screen must answer the most important financial question immediately.
- The graph should feel informative, not decorative.
- The log should read like a ledger: ordered, calm, and trustworthy.
- Empty states should preserve credibility and reinforce the path to earning, not feel playful.

## Implementation Guardrails For Approval

- Preserve the current operational product tone; do not turn these pages into a landing-page style experience.
- Keep section framing consistent across all three pages.
- Reuse semantic colors and tag styles between Lab creation and Collector consumption.
- Do not introduce nested cards inside cards unless there is a clear data-separation reason.
- Ensure mobile retains the same priority order as desktop: action and payout first, detail second.
