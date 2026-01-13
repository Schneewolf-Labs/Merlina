# Merlina Data Editor - Frontend Implementation Plan

## Overview

The Data Editor frontend provides a comprehensive wizard-style interface for transforming raw datasets into ORPO-ready training data. It integrates seamlessly with Merlina's existing wizard theme and provides real-time validation, editing, and transformation capabilities.

---

## 🎨 UI Components

### 1. **Multi-Step Wizard Navigation**

**Location**: Top of interface
**Purpose**: Guide users through the 5-step workflow

**Steps**:
1. **Import** - Upload/select dataset file
2. **Map Schema** - Configure column mappings
3. **Edit & Transform** - Interactive data editing
4. **Validate** - Review quality and fix issues
5. **Export** - Save or upload for training

**Features**:
- Visual progress indicators
- Click to jump between steps
- Validation gates (can't proceed if critical errors)
- Auto-save on step changes

---

### 2. **Import Screen** (Step 1)

**Purpose**: Upload datasets and create editing sessions

**Components**:

#### A. Upload Zone
```html
- Drag & drop area for files
- Click to browse file picker
- Supported formats displayed (JSON, JSONL, CSV, etc.)
- File size limits shown
- Real-time upload progress
```

#### B. Session Configuration
```javascript
{
  sessionName: "My Training Data",
  description: "Dataset for fine-tuning Llama 3...",
  autoDetectSchema: true
}
```

#### C. Import Preview
- Shows first 10 rows after upload
- Displays detected schema type (ORPO, Alpaca, etc.)
- Column statistics (fill rate, data types)
- Suggested mappings preview

**API Calls**:
```javascript
POST /editor/import
  FormData: { file, session_name }

→ Returns: {
  session_id,
  num_rows,
  metadata: { columns, sample, detected_type },
  suggested_mapping
}
```

---

### 3. **Schema Mapper** (Step 2)

**Purpose**: Map source columns to ORPO required fields

**Layout**: Split-panel drag-and-drop interface

#### Left Panel: Source Columns
- Lists all columns from imported data
- Shows sample values on hover
- Color-coded by data type
- Drag handles for mapping

#### Center: Mapping Arrows
- Visual connectors showing mappings
- Template editor for combined fields
- Validation indicators

#### Right Panel: ORPO Fields
```
Required Fields (red if unmapped):
├── prompt (required)
├── chosen (required)
└── rejected (required)

Optional Fields:
├── system
└── reasoning
```

#### Mapping Types

**1. Direct Mapping**
```javascript
{ "prompt": "instruction" }
```

**2. Template Mapping** (combine multiple columns)
```javascript
{
  "prompt": "{instruction}\n\nInput: {input}"
}
```

**3. Nested Path Extraction**
```javascript
{
  "prompt": "messages[0].content",
  "chosen": "messages[1].content"
}
```

**4. Constant Values**
```javascript
{
  "system": "You are a helpful assistant."
}
```

**5. Auto-Generate**
```javascript
{
  "rejected": null,  // Will generate from chosen
  strategy: "truncate_50"
}
```

#### Features:
- Auto-mapping suggestions (pre-filled)
- Live preview of mapped data (5 samples)
- Template editor with syntax highlighting
- Validation warnings for unmapped required fields
- Save/load mapping presets

**API Calls**:
```javascript
POST /editor/transform
  {
    session_id,
    column_mapping: { ... },
    generate_rejected: true,
    rejected_strategy: "truncate_50"
  }

→ Transforms all rows and validates
```

---

### 4. **Table Editor** (Step 3)

**Purpose**: Interactive editing of individual rows

**Features**:

#### A. Data Table
- **Virtual scrolling** for large datasets (1000+ rows)
- **Sortable columns** (click header)
- **Filterable** (search box, status filters)
- **Inline editing** (click cell to edit)
- **Bulk operations** (select multiple rows)
- **Color-coded rows**:
  - Green border: Valid
  - Red background: Has errors
  - Yellow background: Has warnings

#### B. Table Columns
```
| Row # | Status | Prompt | Chosen | Rejected | System | Actions |
|-------|--------|--------|--------|----------|--------|---------|
```

**Cell Renderers**:
- **Long text**: Truncated with "..." and expand button
- **Validation**: Inline error/warning badges
- **Actions**: Edit, Delete, Duplicate buttons

#### C. Action Toolbar
```
Top Toolbar:
├── Add Row
├── Delete Selected
├── Generate Rejected (for selected)
├── Bulk Edit
├── Undo / Redo
└── Save Session

Right Side:
├── Search Box
├── Filter Dropdown (All, Valid, Errors, Warnings)
└── Validate All
```

#### D. Row Editor Modal
Opens when clicking "Edit" or adding new row

```
Modal Layout:
├── Header: Row #123
├── Body:
│   ├── Prompt (textarea, auto-resize)
│   ├── Chosen (textarea, auto-resize)
│   ├── Rejected (textarea, auto-resize)
│   ├── System (textarea, optional)
│   └── Reasoning (textarea, optional)
├── Validation Panel (live):
│   ├── Token counts for each field
│   ├── Similarity % (chosen vs rejected)
│   └── Error/warning list
└── Footer:
    ├── Cancel
    ├── Generate Rejected (dropdown with strategies)
    └── Save
```

**API Calls**:
```javascript
// Add row
POST /editor/session/{id}/row
  { prompt, chosen, rejected, system, reasoning }

// Update row
PUT /editor/session/{id}/row/{idx}
  { prompt, chosen, ... } // Only changed fields

// Delete row
DELETE /editor/session/{id}/row/{idx}

// Undo/Redo
POST /editor/session/{id}/undo
POST /editor/session/{id}/redo

// Generate rejected
POST /editor/generate-pairs/{id}?strategy=truncate_50
```

---

### 5. **Validation Dashboard** (Step 4)

**Purpose**: Review data quality and fix issues

**Layout**:

#### A. Quality Metrics (Top)
```
Cards Grid:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Total Rows  │ Valid Rows  │   Errors    │  Warnings   │
│    847      │  812 (95%)  │     35      │    128      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

#### B. Quality Score Visualization
- Progress bar showing % valid
- Color gradient: Red → Yellow → Green
- Quality score calculation based on errors/warnings

#### C. Statistics Panel
```
Dataset Statistics:
├── Token Length Distribution
│   ├── Prompt: avg 127, max 892, median 98
│   ├── Chosen: avg 342, max 1523, median 287
│   └── Rejected: avg 298, max 1401, median 245
├── Field Completion
│   ├── System: 67% (567/847)
│   └── Reasoning: 12% (102/847)
└── Quality Metrics
    ├── Duplicate prompts: 12
    ├── Very similar pairs: 23
    └── Token count warnings: 45
```

#### D. Issue List (Filterable)
```
Issues found:
┌─ ERROR ──────────────────────────────────────┐
│ Row 1: Missing required field 'rejected'     │
│ [Fix Automatically] [Go to Row]              │
└──────────────────────────────────────────────┘

┌─ WARNING ────────────────────────────────────┐
│ Row 2: Chosen and rejected are 94% similar   │
│ This may provide weak training signal        │
│ [Regenerate Rejected] [Edit Row]             │
└──────────────────────────────────────────────┘

┌─ WARNING ────────────────────────────────────┐
│ Row 15: Prompt is very long (2347 tokens)    │
│ May exceed model context window              │
│ [Edit Row] [Ignore]                          │
└──────────────────────────────────────────────┘
```

**Quick Fix Actions**:
- **Auto-fix** for missing rejected (generate)
- **Regenerate** rejected with different strategy
- **Jump to row** in table editor
- **Batch fix** similar issues

**API Calls**:
```javascript
// Validate entire session
POST /editor/validate/{session_id}

→ Returns: {
  is_valid,
  errors: [...],
  warnings: [...],
  statistics: { ... },
  row_issues: {
    1: { errors: [...], warnings: [...] },
    2: { ... }
  }
}
```

---

### 6. **Preference Pair Generator** (Tab View)

**Purpose**: Generate and compare preference pairs

**Layout**:

#### A. Strategy Selector
```
Dropdown with strategies:
├── Truncate at 30%
├── Truncate at 50% (recommended)
├── Truncate at 70%
├── Degrade formatting
├── Add spelling errors
├── Shuffle sentences
└── Remove details
```

#### B. Preview Mode
Side-by-side comparison of generated pairs

```
┌─── Chosen Response ────┐  ┌─── Rejected Response ───┐
│ Original, high-quality │  │ Truncated/degraded      │
│ response with full     │  │ version with lower      │
│ details...             │  │ quality...              │
└────────────────────────┘  └─────────────────────────┘

           Similarity: 67% ✓ (Good difference)

[Regenerate] [Accept] [Edit Rejected]
```

#### C. Batch Generation
- Select rows (or all incomplete)
- Choose strategy
- Progress indicator
- Review before accepting

**API Calls**:
```javascript
POST /editor/generate-pairs/{session_id}?strategy=truncate_50

→ Generates rejected for all rows missing it
```

---

### 7. **Export Screen** (Step 5)

**Purpose**: Save or deploy the cleaned dataset

**Options**:

#### A. Direct Upload (Recommended)
```
┌─────────────────────────────────────┐
│  📤 Upload to Training Pipeline     │
│                                     │
│  Instantly available for training   │
│  No file download needed            │
│                                     │
│  [Upload for Training] →            │
└─────────────────────────────────────┘
```

#### B. Download File
```
Format Options:
├── JSON (pretty-printed)
├── JSONL (one row per line)
└── CSV

Options:
☑ Export only valid rows (recommended)
☐ Include metadata
☐ Include validation warnings
```

#### C. Save Session
```
💾 Save session for later:
- Continue editing later
- Share with team
- Version control

[Save Session]
```

**Export Preview**:
```json
Preview (first 3 rows):
[
  {
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris...",
    "rejected": "The capital of France is..."
  },
  ...
]

Total: 812 valid rows (35 rows with errors excluded)
```

**API Calls**:
```javascript
POST /editor/export
  {
    session_id,
    format: "json",
    only_valid: true,
    direct_upload: true
  }

→ If direct_upload=true:
  Returns: { dataset_id } // Ready for training

→ If direct_upload=false:
  Returns: { data: [...] } // Download file
```

---

## 🔄 State Management

### Frontend State Structure
```javascript
{
  // Current session
  session: {
    session_id: "uuid",
    name: "My Dataset",
    total_rows: 847,
    column_mapping: { ... },
    statistics: { ... }
  },

  // Current view
  view: {
    currentStep: 2, // 0-4
    currentTab: "table", // table, schema, validation, pairs
    pageSize: 100,
    pageOffset: 0,
    filters: {
      search: "",
      status: "all" // all, valid, errors, warnings
    }
  },

  // Data
  rows: [...], // Current page of rows
  selectedRows: [1, 5, 12], // For bulk operations

  // Validation
  validation: {
    is_valid: false,
    errors: 35,
    warnings: 128,
    row_issues: { ... }
  },

  // UI state
  ui: {
    loading: false,
    saving: false,
    modal: null, // "edit-row", "export", etc.
    editingRow: null,
    toast: null // Success/error messages
  },

  // Undo/redo
  canUndo: true,
  canRedo: false
}
```

---

## 📱 Responsive Design

### Desktop (>1200px)
- Full table view with all columns
- Side-by-side schema mapper
- Multi-panel layout

### Tablet (768px - 1200px)
- Scrollable table
- Stacked panels
- Collapsible sidebars

### Mobile (< 768px)
- Card view instead of table
- Simplified editing (one field at a time)
- Bottom sheet modals
- Touch-friendly controls

---

## ⚡ Performance Optimizations

### 1. Virtual Scrolling
```javascript
// Only render visible rows + buffer
<VirtualTable
  rowHeight={60}
  totalRows={10000}
  visibleRows={20}
/>
```

### 2. Debounced Validation
```javascript
// Validate on edit with 500ms delay
const validateRow = debounce(async (rowIdx) => {
  const result = await api.validateRow(rowIdx);
  updateRowValidation(rowIdx, result);
}, 500);
```

### 3. Pagination
- Load 100 rows at a time
- Lazy load on scroll
- Cache previously loaded pages

### 4. Optimistic Updates
```javascript
// Update UI immediately, sync with server
function updateRow(idx, changes) {
  // Update UI
  dispatch({ type: 'UPDATE_ROW', idx, changes });

  // Sync with server (background)
  api.updateRow(sessionId, idx, changes);
}
```

---

## 🎨 Visual Design System

### Colors
```css
Primary: #764ba2 (Purple)
Success: #28a745 (Green)
Error: #dc3545 (Red)
Warning: #ffc107 (Yellow)
Info: #17a2b8 (Cyan)
```

### Typography
```css
Headings: 'Segoe UI', sans-serif, bold
Body: 'Segoe UI', sans-serif, normal
Code: 'Monaco', 'Courier New', monospace
```

### Spacing
```css
xs: 4px
sm: 8px
md: 16px
lg: 24px
xl: 32px
```

### Animations
```css
Transitions: 0.3s ease
Hover states: transform + shadow
Loading: Pulse animation
Success: Bounce animation
```

---

## 🔧 Technical Stack

### Core Libraries
```javascript
- React 18 (UI framework)
- React Query (API state management)
- Zustand (Local state)
- React Table (Table component)
- React DnD (Drag & drop for schema mapper)
- Monaco Editor (Template editor)
- Chart.js (Statistics visualizations)
- React Toastify (Notifications)
```

### Build Tools
```javascript
- Vite (Fast dev server)
- TypeScript (Type safety)
- ESLint + Prettier (Code quality)
```

### OR: Plain JavaScript Alternative
```javascript
- Vanilla JS with Web Components
- No build step required
- Direct integration with existing Merlina frontend
- Smaller bundle size
```

---

## 📊 User Flows

### Flow 1: Quick Transform (Experienced User)
```
1. Upload file → Auto-detect schema
2. Review suggested mapping → Accept
3. Click "Transform All"
4. Click "Validate All"
5. Click "Upload for Training" → Done!

Time: ~2 minutes
```

### Flow 2: Careful Review (New User)
```
1. Upload file
2. Review samples and columns
3. Manually adjust mappings
4. Generate rejected responses (preview first)
5. Review validation issues
6. Fix errors one by one
7. Export and save session

Time: ~15 minutes
```

### Flow 3: Iterative Editing
```
1. Load existing session
2. Add more rows
3. Edit specific entries
4. Re-validate
5. Export updated dataset

Time: ~5 minutes
```

---

## 🎯 Key Features Summary

✅ **Wizard-style workflow** - Guides users step-by-step
✅ **Auto-schema detection** - Recognizes 6+ dataset formats
✅ **Drag-and-drop mapping** - Visual, intuitive
✅ **Inline editing** - Edit directly in table
✅ **Real-time validation** - Instant feedback
✅ **Preference pair generation** - 7 strategies
✅ **Undo/redo** - Never lose work
✅ **Quality dashboard** - Visual metrics
✅ **Direct export** - One click to training
✅ **Session persistence** - Resume anytime
✅ **Bulk operations** - Scale to large datasets

---

## 📝 Implementation Timeline

### Week 1: Core Infrastructure
- [ ] Setup React/build system
- [ ] Create base wizard component
- [ ] Implement state management
- [ ] API client setup

### Week 2: Import & Schema Mapping
- [ ] Import screen with drag-drop
- [ ] Schema mapper UI
- [ ] Template editor
- [ ] Live preview

### Week 3: Table Editor
- [ ] Virtual scrolling table
- [ ] Inline editing
- [ ] Row editor modal
- [ ] Bulk operations

### Week 4: Validation & Export
- [ ] Validation dashboard
- [ ] Statistics visualizations
- [ ] Issue list with quick fixes
- [ ] Export dialog

### Week 5: Polish & Testing
- [ ] Responsive design
- [ ] Animations & transitions
- [ ] Error handling
- [ ] User testing & fixes

---

## 🚀 Next Steps

1. **Decision**: React-based or Vanilla JS?
2. **Setup**: Initialize frontend project
3. **Design**: Create Figma mockups (optional)
4. **Implement**: Start with Step 1 (Import)
5. **Iterate**: Build and test each component

The backend is ready and waiting! 🎉
