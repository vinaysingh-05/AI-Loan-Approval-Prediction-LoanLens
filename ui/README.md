# UI Layer

The interface is designed as a compact dark decision workspace: clear navigation, a focused form, small supporting imagery, and motion that communicates state without distracting from the prediction.

The Streamlit presentation layer is implemented in the project root `app.py`.

## Navigation

- **Home**: product introduction, model status, and training-data visuals.
- **Prediction**: applicant form, contextual CIBIL/loan charts, decision notification, and confidence gauge.
- **Configuration**: model composition and validation summary.

## Visual assets

The root `assets/` folder stores the visuals used by the interface:

- `outputs/images/pairplot.png`: feature relationship analysis.
- `outputs/images/correlation_heatmap.png`: correlation context from exploration.
- `outputs/images/cibil_score_distribution.png`: CIBIL distribution analysis.
- `outputs/images/loan_amount_distribution.png`: loan amount distribution analysis.
- `assets/approved.jpg`: small approval outcome visual at the top of Home.
- `assets/loan.jpg`: small loan profile visual at the top-right of Prediction.

## Screen map

| Screen | Primary interaction | Image asset | Motion |
| --- | --- | --- | --- |
| Home | Start a prediction | `approved.jpg` | Hero reveal, scan line, breathing status |
| Prediction | Submit applicant form | `loan.jpg` | Card reveal, button lift, result transition |
| Configuration | Inspect model profile | None | Panel reveal |

The image cards use a fixed small width so the form and navigation remain the visual focus. The analysis charts remain available in `outputs/images/` for notebook review but are not loaded into the live page.

Additional brand-ready visuals, `assets/approved.jpg` and `assets/loan.jpg`, are available for future UI sections.

The dark visual system, responsive columns, and CSS motion are kept in `app.py` so the app can run as a single Streamlit entry point.
