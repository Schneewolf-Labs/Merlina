# Merlina Frontend v2.0 - Modular Dashboard Architecture

## Overview

A complete rewrite of the Merlina frontend using a modular, dashboard-based architecture with full support for ORPO and SFT training modes. Built with vanilla JavaScript ES6 modules and modular CSS for maximum maintainability.

## Features

### 🎯 Core Features
- **Dual Training Modes**: Full ORPO and SFT support with mode-aware UI
- **Dashboard Layout**: Grid-based three-column layout (sidebar, main, panel)
- **Real-time Monitoring**: WebSocket-based job updates
- **Responsive Design**: Mobile, tablet, and desktop optimized
- **Modular Architecture**: Clean separation of concerns

### ✨ User Interface
- **Sidebar Navigation**: Quick access to all features
- **Mode Selector**: Visual ORPO vs SFT selection with requirements
- **Stat Cards**: Real-time metrics dashboard
- **Job Panel**: Live training job monitoring
- **Tabbed Configuration**: Organized training settings
- **Dataset Manager**: Preview, validate, and format datasets

### 🎨 Design System
- **CSS Custom Properties**: Comprehensive design tokens
- **Component Library**: Reusable UI components
- **Magical Theme**: Purple gradients, sparkles, animations
- **Accessibility**: ARIA labels, keyboard navigation, focus management

## Architecture

### File Structure

```
frontend/
├── index-v2.html              # Main application shell
│
├── styles/                     # Modular CSS
│   ├── main.css               # Import hub
│   ├── 00-variables.css       # Design tokens
│   ├── 01-reset.css           # CSS reset
│   ├── 02-base.css            # Typography & utilities
│   ├── 03-layout.css          # Grid system
│   ├── 04-components.css      # UI components
│   ├── 05-dashboard.css       # Dashboard styles
│   ├── 06-forms.css           # Form elements
│   └── 07-job-cards.css       # Job monitoring
│
├── js/                         # JavaScript modules
│   ├── app-v2.js              # Main application
│   │
│   ├── core/                   # Core systems
│   │   └── layout-manager.js  # Layout state & responsiveness
│   │
│   ├── managers/               # Business logic
│   │   └── training-mode-manager.js  # ORPO/SFT logic
│   │
│   ├── views/                  # Page views
│   │   ├── dashboard-view.js  # Main dashboard
│   │   ├── training-config-view.js  # Training configuration
│   │   └── dataset-view.js    # Dataset management
│   │
│   ├── components/             # Reusable components
│   │   └── job-panel.js       # Job monitoring panel
│   │
│   └── services/               # API communication
│       └── api-client.js      # HTTP & WebSocket clients
│
└── assets/                     # Static assets
    └── images/
```

### Component Hierarchy

```
MerlinaAppV2 (Main App)
├── LayoutManager (Grid layout, navigation)
├── TrainingModeManager (ORPO/SFT logic)
├── JobPanel (Real-time job monitoring)
│
├── Views
│   ├── DashboardView (Landing page)
│   ├── TrainingConfigView (Training setup)
│   ├── DatasetView (Dataset configuration)
│   └── [Other views - placeholders]
│
└── API Clients
    ├── TrainingAPI (Job management)
    ├── DatasetAPI (Dataset operations)
    ├── GPUAPI (GPU monitoring)
    ├── ModelAPI (Model validation)
    └── StatsAPI (Application stats)
```

## Usage

### Accessing the Application

1. **Start Merlina backend:**
   ```bash
   python merlina.py
   ```

2. **Open in browser:**
   ```
   http://localhost:8000/index-v2.html
   ```

### Navigation

- **Sidebar**: Click any item to navigate between views
- **Dashboard**: Click action cards for quick navigation
- **Keyboard Shortcuts**:
  - `Ctrl/Cmd + /`: Toggle sidebar
  - `Esc`: Close overlays
  - Arrow keys: Navigate within views

### Training Workflow

1. **Select Mode**: Choose ORPO or SFT on dashboard
2. **Configure Dataset**: Load and preview your dataset
3. **Setup Training**: Configure parameters with tabs
4. **Monitor Jobs**: Watch live progress in right panel
5. **Review Results**: Check completed jobs and metrics

## Key Components

### LayoutManager

Handles application layout and responsiveness.

```javascript
import { LayoutManager } from './core/layout-manager.js';

const layout = new LayoutManager();
layout.toggleSidebar(true);  // Show sidebar
layout.togglePanel(false);   // Hide right panel
layout.navigateToView('dashboard');
```

### TrainingModeManager

Manages ORPO vs SFT training modes.

```javascript
import { TrainingModeManager } from './managers/training-mode-manager.js';

const modeManager = new TrainingModeManager();
modeManager.setMode('sft');  // Switch to SFT
modeManager.getRequiredColumns();  // ['prompt', 'chosen']
modeManager.validateDataset(['prompt', 'chosen']);  // {valid: true}
```

### JobPanel

Real-time job monitoring with WebSocket updates.

```javascript
import { JobPanel } from './components/job-panel.js';

const panel = new JobPanel();
panel.addJob(jobData);       // Add active job
panel.updateJob(id, data);   // Update with new data
panel.removeJob(id);         // Remove completed job
```

### API Clients

HTTP and WebSocket communication with backend.

```javascript
import { trainingAPI, datasetAPI } from './services/api-client.js';

// Submit training job
const result = await trainingAPI.submitJob(config);

// Preview dataset
const preview = await datasetAPI.previewRaw(datasetConfig);

// Get job status
const status = await trainingAPI.getJobStatus(jobId);
```

## Customization

### Adding a New View

1. **Create view file:**
   ```javascript
   // js/views/my-view.js
   export class MyView {
       render() {
           return `<div class="card">...</div>`;
       }
       attachEventListeners() {
           // Add event handlers
       }
   }
   ```

2. **Register in app:**
   ```javascript
   // js/app-v2.js
   import { MyView } from './views/my-view.js';

   registerViews() {
       this.views.set('my-view', new MyView());
   }
   ```

3. **Add to sidebar:**
   ```html
   <!-- index-v2.html -->
   <div class="sidebar-nav-item" data-view="my-view">
       <span class="sidebar-nav-icon">🎨</span>
       <span class="sidebar-nav-text">My View</span>
   </div>
   ```

### Custom Styling

Override CSS variables in your own stylesheet:

```css
:root {
    --primary-purple: #your-color;
    --sidebar-width: 280px;
    --panel-width: 400px;
}
```

### Adding API Endpoints

Extend the API clients:

```javascript
// js/services/api-client.js
export class MyAPI extends APIClient {
    async customEndpoint(data) {
        return this.post('/my-endpoint', data);
    }
}
```

## Development

### Code Style

- **ES6 Modules**: Use import/export
- **Async/Await**: For asynchronous operations
- **JSDoc Comments**: Document all public methods
- **Consistent Naming**: camelCase for variables, PascalCase for classes

### File Organization

- **One component per file**: Each view/component in separate file
- **Group by feature**: Related files in same directory
- **Clear naming**: Descriptive file and function names

### Testing

Test in multiple browsers:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Test responsive breakpoints:
- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: 1024px - 1280px
- Wide: > 1280px

## Performance

### Optimization Tips

1. **Lazy Loading**: Views loaded only when navigated to
2. **Event Delegation**: Minimize event listeners
3. **Debouncing**: Used for resize and input events
4. **CSS Containment**: Layout containment for better rendering
5. **WebSocket Pooling**: Reuse connections where possible

### Bundle Size

- **No Build Tools**: Zero build process overhead
- **Modular Loading**: Browser-native ES6 modules
- **CSS Imports**: Modular CSS with @import
- **Total Size**: ~50KB uncompressed (CSS + JS)

## Browser Support

- **Modern Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
- **ES6 Modules**: Native module support required
- **CSS Grid**: Full grid support required
- **WebSockets**: For real-time updates
- **No IE11**: Not supported (End of Life)

## Accessibility

- **Keyboard Navigation**: Full keyboard support
- **Screen Readers**: ARIA labels and semantic HTML
- **Focus Management**: Proper focus indicators
- **Skip Links**: Skip to main content
- **Color Contrast**: WCAG AA compliant
- **Reduced Motion**: Respects prefers-reduced-motion

## Migration from v1

### Key Differences

| Feature | v1 (Original) | v2 (Modular) |
|---------|--------------|--------------|
| Layout | Single column scroll | Three-column grid |
| Navigation | Scroll-based steps | Sidebar navigation |
| Mode Selection | Dropdown | Visual cards |
| Job Monitoring | Modal popup | Side panel |
| CSS | Single file | 8 modular files |
| JS | Monolithic | ES6 modules |
| Responsiveness | Basic | Full responsive |

### Backward Compatibility

- Original frontend still available at `/index.html`
- v2 frontend at `/index-v2.html`
- Both share same backend API
- LocalStorage keys compatible

## Troubleshooting

### Common Issues

**Q: Blank screen on load**
- Check browser console for errors
- Ensure ES6 module support
- Verify all files are served correctly

**Q: Styles not loading**
- Check CSS @import paths
- Verify MIME types (text/css)
- Clear browser cache

**Q: WebSocket connection fails**
- Check backend is running
- Verify WebSocket endpoint
- Check browser dev tools Network tab

**Q: Mode switching doesn't work**
- Check TrainingModeManager initialization
- Verify event listeners attached
- Check browser console for errors

## Future Enhancements

### Planned Features

- [ ] Job history with pagination
- [ ] Advanced analytics charts
- [ ] Configuration templates library
- [ ] Drag-and-drop dataset upload
- [ ] Multi-language support
- [ ] Dark mode toggle
- [ ] Export/import configurations
- [ ] Collaborative training queue

### Performance Improvements

- [ ] Service Worker for offline support
- [ ] IndexedDB for local caching
- [ ] Virtual scrolling for long lists
- [ ] Progressive Web App features

## Contributing

When contributing to the v2 frontend:

1. Follow existing code style
2. Add JSDoc comments
3. Test on multiple browsers
4. Maintain modular architecture
5. Update this README if needed

## License

Same as Merlina project.

## Credits

Built with ✨ for the Merlina LLM training system.
- Architecture: Modular ES6 + CSS Grid
- Theme: Magical wizard aesthetic
- Purpose: Making ML accessible and delightful
