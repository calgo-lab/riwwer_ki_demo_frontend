"""Static header rendering for RIWWER ML Demo."""
import os
import streamlit as st


def render_static_header() -> None:
    """Render static header elements (logos, intro, CSS). Runs once at startup."""
    
    # Logos row (uniform height, concatenated with fixed spacing)
    try:
        from pathlib import Path
        import base64

        def _b64_img(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")

        logos = [
            "figures/riwwer-logo.png",
            "figures/bht-logo.png",
            "figures/calgolab-logo.png",
            "figures/okeanos-logo.png",
            "figures/ude-logo.png",
        ]
        logo_files = [p for p in logos if os.path.exists(p)]

        if logo_files:
            logo_h = 100  # px fixed height
            gap_px = 24   # fixed horizontal spacing
            # CSS for a single-row centered flex layout with fixed gaps
            st.markdown(
                f"""
                <style>
                .logo-row {{
                    display: flex;
                    align-items: center;
                    justify-content: left;
                    gap: {gap_px}px;
                    flex-wrap: nowrap;
                    margin-bottom: 8px;
                }}
                .logo-row img {{
                    height: {logo_h}px;
                    object-fit: contain;
                    display: inline-block;
                }}
                @media (max-width: 900px) {{
                    .logo-row {{ flex-wrap: wrap; gap: 16px; }}
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Build the row of images
            imgs_html = []
            for path in logo_files:
                try:
                    b64 = _b64_img(path)
                    imgs_html.append(
                        f"<img src='data:image/png;base64,{b64}' alt='{os.path.basename(path)}' />"
                    )
                except Exception:
                    imgs_html.append(
                        f"<img src='{path}' alt='{os.path.basename(path)}' />"
                    )

            st.markdown("<div class='logo-row'>" + "".join(imgs_html) + "</div>", unsafe_allow_html=True)
    except Exception:
        pass

    # Short intro text (full-width, styled for readability)
    # Pick text color depending on Streamlit theme (white on dark theme)
    # Theme-aware styling
    try:
        current_theme = st.context.theme.type
    except Exception:
        current_theme = 'light'
    is_dark = (current_theme == 'dark')
    _intro_text_color = "#ffffff" if is_dark else "#111111"
    st.markdown(
        f"""
        <div style="width:100%; box-sizing:border-box; padding:0 16px; font-size:19px; line-height:1.45; color:{_intro_text_color};">
        <strong>Welcome to the RIWWER ML Demo!</strong> This application showcases the machine learning (ML) models for Urban Wastewater Management
        developed by the Berliner Hochschule für Technik, Okeanos and the University of Duisburg-Essen. 
        The models are applied to historical data from the combined sewer system of Vierlinden in Duisburg (<em>Wirtschaftsbetriebe Duisburg</em>).
        We demonstrate the performance of ML models to forecast filling levels and estimate the risk of Combined Sewer Overflows in the year 2023.
        The models were trained with data from the years of 2021 and 2022. For further information consult our GitHub repository: <a href="https://github.com/calgo-lab/resilient-timeseries-evaluation">resilient-timeseries-evaluation</a><br/>
        <br/>
        Start by navigating through time using the buttons and sliders in the <strong>Time Navigation Control</strong>. Alternatively you can also search for specific rain events using the <em>"Select rainfall"</em> slider.<br/>
        <br/>
        The project was funded by the Federal Ministry of Economic Affairs and Climate Action of Germany for the RIWWER project (01MD22007H, 01MD22007C).<br/>
        <em>RIWWER: Reduction of the Impact of untreated WasteWater on the Environment in case of torrential Rain</em>
        <br/><br/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Styles for bordered panels with a blue title bar
    st.markdown(
        """
        <style>
        .panel-header {
            background: #c2cbfc;
            border-left: 4px solid #1f6feb;
            color: #0b63ce;
            padding: 0.4rem 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Demo video & citation rendered in the static header to avoid being re-created by fragments
    try:
        video_url = "https://cloud.bht-berlin.de/public.php/dav/files/b9xt4T3SdiLBiFZ"
        # Put Demo Video and Cite Us side-by-side in columns (4:2)
        cols = st.columns([5, 4], gap="small")
        # Left: embedded video
        with cols[0]:
            with st.container(border=True):
                st.markdown("<div class='panel-header'>🎬 Demo Video</div>", unsafe_allow_html=True)
                try:
                    html5_video = (
                        f'<div style="display:flex;justify-content:center;">'
                        f'<div style="width:min(100%, 75rem);">'
                        f'<video controls preload="metadata" style="display:block;width:100%;height:auto;aspect-ratio:16/9;border-radius:8px;" src="{video_url}">'
                        f'Your browser does not support the video tag.'
                        f'</video>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(html5_video, unsafe_allow_html=True)
                except Exception:
                    st.markdown(
                        f"<div style='text-align:center; padding:12px 8px;'><a href='{video_url}' target='_blank' style='display:inline-block; padding:10px 18px; background:#1f6feb; color:#fff; border-radius:6px; text-decoration:none; font-weight:600;'>▶️ Open Video</a></div>",
                        unsafe_allow_html=True,
                    )
        # Right: citation and links with matching font size to intro
        with cols[1]:
            with st.container(border=True):
                st.markdown("<div class='panel-header'>📚 Learn More & Cite Us</div>", unsafe_allow_html=True)
                pub_html = f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'>\n<strong>Publication:</strong><br/>\n\"A Resilient Solution for Sewer Overflow Monitoring across Cloud and Edge\"\n</div>"
                st.markdown(pub_html, unsafe_allow_html=True)

                citation_text = '''@article{singh2026resilientsolutionseweroverflow,
    title={A Resilient Solution for Sewer Overflow Monitoring across Cloud and Edge}, 
    author={Vipin Singh and Tianheng Ling and Peter Ghaly and Felix Grimmeisen and Gregor Schiele and Felix Biessmann},
    year={2026},
    eprint={2605.10592},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2605.10592}, 
}'''

                st.markdown(f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'><strong>BibTeX Citation:</strong></div>", unsafe_allow_html=True)            
                st.code(citation_text, language="bibtex")
                st.markdown(f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'><strong>Paper URL (arXiv):</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:19px; line-height:1.45;'><a href='https://arxiv.org/abs/2605.10592' target='_blank'>https://arxiv.org/abs/2605.10592</a></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:19px; line-height:1.45; color:{_intro_text_color};'>accepted at 35th International Joint Conference on Artificial Intelligence 2026 (IJCAI-ECAI 2026), Demonstrations Track.</div>", unsafe_allow_html=True)
    except Exception:
        # Avoid breaking the static header if any of these elements fail
        pass