import os
import io
import base64
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, ctx
import glob
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate


# === Default Parameter Settings ===
default_r, default_t, default_m = 15, 2, 7

# === Load Default Data ===
default_file = "./files/demo.xlsx"
default_df = pd.read_excel(default_file)
default_df.columns = ["r", "t", "m", "correct_matches", "total_matches", "time"]
r_values = sorted(default_df["r"].unique())
t_values = sorted(default_df["t"].unique())
m_values = sorted(default_df["m"].unique())
correct_password = "123"

# === Initialize Dash App ===
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="GIMS Parameter Analyzer"
)

# === Layout ===
app.layout = html.Div([
    dcc.Store(id="auth-status", data=False),
    html.Div(id="auth-container")
])

MAIN_CONTENT = html.Div([
    dcc.Store(id="loading-flag", data=False),
    html.Div(id="loading-message", style={
        "position": "fixed",
        "top": "10px",
        "left": "50%",
        "transform": "translateX(-50%)",
        "backgroundColor": "#fffae6",
        "border": "1px solid #ffe58f",
        "padding": "10px 20px",
        "borderRadius": "6px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
        "display": "none",
        "zIndex": 10000,
        "fontSize": "16px",
        "color": "#ad8b00"
    }),

    html.H1("GIMS Parameter Analysis Dashboard", style={
        "textAlign": "center",
        "margin": "30px auto 20px auto",
        "fontSize": "36px",
        "color": "#333"
    }),
    html.Div([
        html.B("Parameter Description:"),
        html.Ul([
            html.Li([
                html.B("r"),
                ": Radius threshold — determines which keypoints are considered neighbors during graph construction. ",
                "Larger values create denser graphs. In real measurements, a large r increases matching quantity but may introduce noise, ",
                "while a small r ensures sparsity but risks missing valid matches."
            ]),
            html.Li([
                html.B("t"),
                ": Similarity percentile — controls edge filtering by thresholding on similarity scores. ",
                "Higher t makes the graph sparser and more selective. ",
                "Experimentally, low t tends to preserve more matches but lowers overall accuracy due to weak connections."
            ]),
            html.Li([
                html.B("m"),
                ": Minimum subgraph size — filters out small disconnected subgraphs. ",
                "In practice, increasing m significantly improves robustness by discarding noisy fragments, ",
                "especially in complex scenes with occlusion or clutter."
            ])
        ]),
        html.Details([
            html.Summary("\ud83d\udcd8 Parameter Analysis (click to expand)"),
            dcc.Markdown('''
                **r — Radius Threshold**
                - Controls which keypoints are considered neighbors during graph construction.
                - Larger values increase connectivity and matching quantity but may introduce noise and raise computation time.
                - Smaller values enforce sparsity, reducing noise but risking missing valid connections.

                **t — Similarity Percentile**
                - Filters weak edges based on feature similarity (e.g., cosine similarity).
                - Higher `t` removes more low-confidence edges, improving robustness.
                - Lower `t` retains more matches but may dilute edge quality.

                **m — Minimum Subgraph Size**
                - Removes small disconnected subgraphs after edge pruning.
                - Higher `m` filters noisy fragments, improves matching stability.
                - Lower `m` keeps all components but risks clutter from unstable structures.
            ''')
        ])
    ], style={
        "fontSize": "16px",
        "padding": "10px",
        "border": "1px solid #ccc",
"boxShadow": "0 4px 12px rgba(0, 0, 0, 0.15)",
"transition": "box-shadow 0.3s ease-in-out",
        "borderRadius": "12px",
        "backgroundColor": "#f9f9f9",
        "width": "80%",
        "margin": "auto"
    }),

    html.Div([
        html.B("File Loader:"),
        html.Label("Upload a new Excel file (.xlsx):", style={"fontSize": "16px", "display": "block", "width": "90%", "margin": "0 auto 10px auto"}),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and drop or ", html.A("click to upload")]),
            style={
                "width": "90%", "height": "60px", "lineHeight": "60px",
                "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                "textAlign": "center", "margin": "10px auto", "backgroundColor": "#fafafa"
            },
            multiple=False
        ),
        html.Label("Or select a file from ./files directory:", style={"fontSize": "16px", "display": "block", "width": "90%", "margin": "20px auto 5px auto"}),
        dcc.Dropdown(
            id="file-dropdown",
            options=[
                {"label": os.path.relpath(f, "files"), "value": os.path.relpath(f, "files")}
                for f in sorted(glob.glob("files/*/*.xlsx"))
            ],
            placeholder="Select an Excel file from subfolder",
            style={"width": "95%", "margin": "0 auto 10px auto"}
        ),
        html.Div(id="filename-display", children=f"✅ Current file: {os.path.basename(default_file)}", style={
            "textAlign": "center", "marginTop": "10px", "fontSize": "16px", "color": "#444"
        })
    ], style={
        "fontSize": "16px",
        "padding": "10px",
        "border": "2px solid #ccc",
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
        "borderRadius": "10px",
        "backgroundColor": "#f0f9ff",
        "width": "80%",
        "margin": "20px auto"
    }),
    dcc.Store(id="data-store", data=default_df.to_json(date_format='iso', orient='split')),
    html.Div(id="image-preview", style={ "width": "88%", "textAlign": "center", "margin": "0 auto"}),
    dcc.Store(id="lightbox-image-src"),  # 用于存储点击图片的base64路径
    html.Div(id="lightbox-modal-container"),  # 容纳动态生成的 Modal 弹窗
    html.Div([
        html.Button("\ud83d\udd04 Reset All Sliders", id="reset-button", n_clicks=0, title="Click to reset all sliders", style={
            "backgroundColor": "#007BFF",
            "color": "white",
            "fontSize": "16px",
            "padding": "10px 20px",
            "border": "none",
            "borderRadius": "8px",
            "cursor": "pointer",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
            "margin": "20px auto",
            "display": "block",
            "textAlign": "center",
            "width": "200px"
        }),
        html.Div(
            id="reset-feedback",
            style={
                "position": "fixed",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "padding": "12px 20px",
                "backgroundColor": "#d4edda",
                "color": "#155724",
                "border": "1px solid #c3e6cb",
                "borderRadius": "8px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.15)",
                "zIndex": 9999,
                "fontSize": "14px",
                "display": "none"
            }
        ),
        dcc.Interval(id="reset-feedback-timer", interval=2000, n_intervals=0, disabled=True)
    ], style={"textAlign": "center"}),

    html.Div([
        html.H3("Joint Parameter Selection (r, t, m Heatmaps)", style={"textAlign": "center", "marginBottom": "20px"}),

        html.Div([
            html.Label("Select r parameter (show t × m heatmap):"),
            dcc.Slider(
                id="r-slider", min=min(r_values), max=max(r_values), step=1,
                marks={int(r): str(r) for r in r_values}, value=default_r,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([
                dcc.Graph(id="heatmap-correct", style={"width": "45%", "display": "inline-block"}),
                dcc.Graph(id="heatmap-time", style={"width": "45%", "display": "inline-block"})
            ], style={"display": "flex", "justifyContent": "center", "gap": "100px"})
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("Select t parameter (show r × m heatmap):"),
            dcc.Slider(
                id="t-slider", min=min(t_values), max=max(t_values), step=1,
                marks={int(t): str(t) for t in t_values}, value=default_t,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([
                dcc.Graph(id="heatmap-rm-correct", style={"width": "45%", "display": "inline-block"}),
                dcc.Graph(id="heatmap-rm-time", style={"width": "45%", "display": "inline-block"})
            ], style={"display": "flex", "justifyContent": "center", "gap": "100px"})
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("Select m parameter (show r × t heatmap):"),
            dcc.Slider(
                id="m-slider", min=min(m_values), max=max(m_values), step=1,
                marks={int(m): str(m) for m in m_values}, value=default_m,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([
                dcc.Graph(id="heatmap-rt-correct", style={"width": "45%", "display": "inline-block"}),
                dcc.Graph(id="heatmap-rt-time", style={"width": "45%", "display": "inline-block"})
            ], style={"display": "flex", "justifyContent": "center", "gap": "100px"})
        ])
    ], style={
        "padding": "10px",
        "margin": "30px auto",
        "border": "2px solid #ccc",
        "borderRadius": "12px",
        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
        "backgroundColor": "#f9f9ff",
        "width": "80%"
    }),

    html.Div([
        html.H3("3D Parameter Visualization (r-t-m)", style={"textAlign": "center", "marginBottom": "20px"}),
        html.Div([
            dcc.Graph(id="scatter3d-correct", style={"width": "49%", "display": "inline-block"}),
            dcc.Graph(id="scatter3d-time", style={"width": "49%", "display": "inline-block"})
        ])
    ], style={
        "padding": "10px",
        "margin": "20px auto",
        "border": "1px solid #ccc",
        "borderRadius": "10px",
        "width": "80%",
        "backgroundColor": "#f9f9ff"
    }),

    html.Div([
        html.H3("Univariate Analysis (interactive sliders to fix other parameters)", style={"textAlign": "center", "marginBottom": "20px"}),
        html.Div([
            html.Label("Select t (for r-axis analysis):"),
            dcc.Slider(id="line-r-t-slider", min=min(t_values), max=max(t_values), step=1,
                       marks={int(t): str(t) for t in t_values}, value=default_t),
            html.Label("Select m (for r-axis analysis):"),
            dcc.Slider(id="line-r-m-slider", min=min(m_values), max=max(m_values), step=1,
                       marks={int(m): str(m) for m in m_values}, value=default_m),
            html.Div([
                dcc.Graph(id="line-r-correct", style={"width": "45%"}),
                dcc.Graph(id="line-r-time", style={"width": "45%"})
            ], style={"display": "flex", "justifyContent": "center", "gap": "20px"})
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Select r (for t-axis analysis):"),
            dcc.Slider(id="line-t-r-slider", min=min(r_values), max=max(r_values), step=1,
                       marks={int(r): str(r) for r in r_values}, value=default_r),
            html.Label("Select m (for t-axis analysis):"),
            dcc.Slider(id="line-t-m-slider", min=min(m_values), max=max(m_values), step=1,
                       marks={int(m): str(m) for m in m_values}, value=default_m),
            html.Div([
                dcc.Graph(id="line-t-correct", style={"width": "45%"}),
                dcc.Graph(id="line-t-time", style={"width": "45%"})
            ], style={"display": "flex", "justifyContent": "center", "gap": "20px"})
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Select r (for m-axis analysis):"),
            dcc.Slider(id="line-m-r-slider", min=min(r_values), max=max(r_values), step=1,
                       marks={int(r): str(r) for r in r_values}, value=default_r),
            html.Label("Select t (for m-axis analysis):"),
            dcc.Slider(id="line-m-t-slider", min=min(t_values), max=max(t_values), step=1,
                       marks={int(t): str(t) for t in t_values}, value=default_t),
            html.Div([
                dcc.Graph(id="line-m-correct", style={"width": "45%"}),
                dcc.Graph(id="line-m-time", style={"width": "45%"})
            ], style={"display": "flex", "justifyContent": "center", "gap": "20px"})
        ])
    ], style={
        "padding": "10px",
        "margin": "20px auto",
        "border": "1px solid #ccc",
        "borderRadius": "10px",
        "width": "80%",
        "backgroundColor": "#fff9f9"
    })
])

# === Callbacks ===
@app.callback(
    Output("auth-container", "children"),
    Input("auth-status", "data")
)
def render_content(authenticated):
    if not authenticated:
        return html.Div([
            html.H2("🔐 Access Restricted", style={"textAlign": "center", "marginTop": "60px"}),
            html.Div([
                "Please enter the password to access the dashboard. Hint: The password is ",
                html.Span("123", style={"color": "red", "fontWeight": "bold"})
            ], style={"textAlign": "center", "marginBottom": "20px", "color": "#888"}),
            dcc.Input(id="password-input", type="password", placeholder="Enter password here...",
                      style={"margin": "auto", "display": "block", "padding": "10px", "fontSize": "16px",
                             "width": "300px", "border": "1px solid #ccc", "borderRadius": "6px"}),
            html.Button("Submit", id="submit-password", n_clicks=0,
                        style={"display": "block", "margin": "20px auto", "padding": "10px 20px",
                               "fontSize": "16px", "borderRadius": "6px", "backgroundColor": "#007BFF",
                               "color": "white", "border": "none", "cursor": "pointer"}),
            html.Div(id="password-feedback", style={"textAlign": "center", "color": "red", "marginTop": "10px"})
        ])
    else:
        # 返回主页面 layout 内容（你已有的主页面布局）
        return MAIN_CONTENT


@app.callback(
    Output("auth-status", "data"),
    Output("password-feedback", "children"),
    Input("submit-password", "n_clicks"),
    State("password-input", "value"),
    prevent_initial_call=True
)
def check_password(n_clicks, input_password):
    if input_password == correct_password:
        return True, ""
    return False, "❌ Incorrect password. Please try again."


@app.callback(
    Output("data-store", "data"),
    Output("filename-display", "children"),
    Output("image-preview", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    Input("file-dropdown", "value"),
    prevent_initial_call=True
)
def load_data(upload_contents, upload_filename, dropdown_filename):
    triggered_id = ctx.triggered_id
    if triggered_id == "upload-data" and upload_contents is not None:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        filename = upload_filename
    elif triggered_id == "file-dropdown" and dropdown_filename is not None:
        filepath = os.path.join("files", dropdown_filename)
        df = pd.read_excel(filepath)
        filename = dropdown_filename
        # 查找图片（jpg/png）
        folder = os.path.dirname(filepath)
        image_html = []
        image_output = None
        for ext in ("*.jpg", "*.png", "*.bmp"):
            for image_path in sorted(glob.glob(os.path.join(folder, ext))):
                encoded = base64.b64encode(open(image_path, "rb").read()).decode()
                img_tag = html.Img(
                    id={"type": "image-thumb", "index": image_path},
                    src=f"data:image/{ext[2:]};base64,{encoded}",
                    style={
                        "width": "240px",
                        "margin": "10px",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
                        "objectFit": "cover",
                        "cursor": "pointer",
                        "transition": "transform 0.2s",
                    },
                    n_clicks=0
                )
                image_html.append(img_tag)
                image_output = html.Div([
                        html.H4("📷 Related Images", style={
                            "textAlign": "center",
                            "marginTop": "0px",
                            "marginBottom": "20px",
                            "color": "#333"
                        }),
                        html.Div(image_html, style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "justifyContent": "center",
                            "gap": "100px",  # 图片之间的间距
                        })
                        ], style={
                            "padding": "20px",
                            "margin": "20px auto",
                            "width": "89%",
                            "border": "1px solid #ddd",
                            "borderRadius": "12px",
                            "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
                            "backgroundColor": "#fdfdfd"
                        }) if image_html else None
    else:
        return dash.no_update, dash.no_update, None
    df.columns = ["r", "t", "m", "correct_matches", "total_matches", "time"]
    display_name = f"✅ Current file: {filename}"
    return df.to_json(date_format='iso', orient='split'), f"✅ Current file: {filename}", image_output


# --------------------image-preview------------------------ #
@app.callback(
    Output("lightbox-image-src", "data"),
    Input({"type": "image-thumb", "index": ALL}, "n_clicks"),
    State({"type": "image-thumb", "index": ALL}, "src"),
    prevent_initial_call=True
)
def show_image_modal(n_clicks_list, src_list):
    if not any(n_clicks_list):
        return dash.no_update
    clicked_index = n_clicks_list.index(max(n_clicks_list))
    return src_list[clicked_index]

@app.callback(
    Output("lightbox-modal-container", "children"),
    Input("lightbox-image-src", "data"),
    prevent_initial_call=True
)
def display_modal(image_src):
    if not image_src:
        return None
    return html.Div([
        html.Div([
            html.Img(src=image_src, style={
                "maxWidth": "90vw",
                "maxHeight": "90vh",
                "borderRadius": "12px",
                "boxShadow": "0 6px 16px rgba(0,0,0,0.3)"
            }),
            html.Div("Click anywhere to close", style={
                "marginTop": "10px",
                "color": "#fff",
                "fontSize": "14px"
            })
        ], style={"textAlign": "center"})
    ], style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100vw",
        "height": "100vh",
        "backgroundColor": "rgba(0, 0, 0, 0.75)",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "zIndex": 9999,
        "cursor": "pointer"
    }, id="modal-overlay", n_clicks=0)

@app.callback(
    Output("lightbox-image-src", "data", allow_duplicate=True),
    Input("modal-overlay", "n_clicks"),
    prevent_initial_call=True
)
def close_modal(n):
    return None

@app.callback(
    Output("loading-message", "children"),
    Output("loading-message", "style"),
    Input("loading-flag", "data")
)
def display_loading_message(is_loading):
    style_base = {
        "position": "fixed",
        "top": "10px",
        "left": "50%",
        "transform": "translateX(-50%)",
        "backgroundColor": "#fffae6",
        "border": "1px solid #ffe58f",
        "padding": "10px 20px",
        "borderRadius": "6px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
        "zIndex": 10000,
        "fontSize": "16px",
        "color": "#ad8b00"
    }
    if is_loading:
        return "⏳ Loading data and rendering visualizations...", {**style_base, "display": "block"}
    return "", {**style_base, "display": "none"}

@app.callback(
    Output("loading-flag", "data"),
    Input("file-dropdown", "value"),
    prevent_initial_call=True
)
def flag_loading_on_dropdown_change(file_value):
    if file_value:
        return True
    return dash.no_update


# ----------------------------------------------------------- #

@app.callback(
    Output("heatmap-correct", "figure"),
    Output("heatmap-time", "figure"),
    Output("loading-flag", "data", allow_duplicate=True),
    Input("r-slider", "value"),
    Input("data-store", "data"),
    prevent_initial_call=True
)
def update_r(r_val, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    df_r = df[df["r"] == r_val]
    pivot_match = df_r.pivot_table("correct_matches", index="t", columns="m", aggfunc="mean")
    pivot_time = df_r.pivot_table("time", index="t", columns="m", aggfunc="mean")
    figs = []
    for pivot, title, cmap in zip(
        [pivot_match, pivot_time],
        ["Correct Matches", "Time"],
        ["YlGnBu", "YlOrRd"]
    ):
        fig = px.imshow(pivot, text_auto=False, color_continuous_scale=cmap, title=title)
        fig.update_layout(
            xaxis_title="m", yaxis_title="t", height=450, title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(x=0.8, len=1, title=None),
        )
        fig.add_vline(x=default_m, line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=default_t, line_dash="dash", line_color="red", line_width=2)
        figs.append(fig)
    return figs[0], figs[1], False

@app.callback(
    Output("heatmap-rm-correct", "figure"),
    Output("heatmap-rm-time", "figure"),
    Input("t-slider", "value"),
    Input("data-store", "data")
)
def update_t(t_val, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    df_t = df[df["t"] == t_val]
    pivot_match = df_t.pivot_table("correct_matches", index="r", columns="m", aggfunc="mean")
    pivot_time = df_t.pivot_table("time", index="r", columns="m", aggfunc="mean")
    figs = []
    for pivot, title, cmap in zip(
        [pivot_match, pivot_time],
        ["Correct Matches", "Time"],
        ["YlGnBu", "YlOrRd"]
    ):
        fig = px.imshow(pivot, text_auto=False, color_continuous_scale=cmap, title=title)
        fig.update_layout(xaxis_title="m", yaxis_title="r", height=450, title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(x=0.8, len=1, title=None),
        )
        fig.add_vline(x=default_m, line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=default_r, line_dash="dash", line_color="red", line_width=2)
        figs.append(fig)
    return figs[0], figs[1]


@app.callback(
    Output("heatmap-rt-correct", "figure"),
    Output("heatmap-rt-time", "figure"),
    Input("m-slider", "value"),
    Input("data-store", "data")
)
def update_m(m_val, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    df_m = df[df["m"] == m_val]
    pivot_match = df_m.pivot_table("correct_matches", index="r", columns="t", aggfunc="mean")
    pivot_time = df_m.pivot_table("time", index="r", columns="t", aggfunc="mean")

    figs = []
    for pivot, title, cmap in zip(
        [pivot_match, pivot_time],
        ["Correct Matches", "Time"],
        ["YlGnBu", "YlOrRd"]
    ):
        fig = px.imshow(pivot, text_auto=False, color_continuous_scale=cmap, title=title)
        fig.update_layout(xaxis_title="t", yaxis_title="r", height=450, title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(x=0.8, len=1, title=None),
        )
        fig.add_vline(x=default_t, line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=default_r, line_dash="dash", line_color="red", line_width=2)
        figs.append(fig)
    return figs[0], figs[1]


@app.callback(
    Output("scatter3d-correct", "figure"),
    Output("scatter3d-time", "figure"),
    Input("data-store", "data")
)
def update_3d(data):
    df = pd.read_json(io.StringIO(data), orient='split')
    # Correct Matches 图
    fig_correct = px.scatter_3d(
        df, x="r", y="t", z="m",
        color="correct_matches",
        hover_data={
            "r": True, "t": True, "m": True,
            "correct_matches": True,
            "time": True
        },
        title="Correct Matches"
    )
    fig_correct.update_traces(marker=dict(size=3, opacity=0.8))
    fig_correct.update_layout(
        title_x=0.5, title_y=0, title_yanchor='bottom',
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(x=0.7, len=0.7, title=None),
        scene=dict(
            yaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
        )
    )
    # Time 图
    fig_time = px.scatter_3d(
        df, x="r", y="t", z="m",
        color="time",
        hover_data={
            "r": True, "t": True, "m": True,
            "correct_matches": True,
            "time": True
        },
        title="Time"
    )
    fig_time.update_traces(marker=dict(size=3, opacity=0.8))
    fig_time.update_layout(
        title_x=0.5, title_y=0, title_yanchor='bottom',
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(x=0.7, len=0.7, title=None),
        scene=dict(
            yaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
        )
    )
    return fig_correct, fig_time


@app.callback(
    Output("scatter3d-time", "figure", allow_duplicate=True),
    Input("scatter3d-correct", "relayoutData"),
    State("data-store", "data"),
    prevent_initial_call=True
)
def sync_camera(correct_camera, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    fig_time = px.scatter_3d(df, x="r", y="t", z="m", color="time", title="Time")
    fig_time.update_traces(marker=dict(size=3, opacity=0.8))
    camera_eye = dict(x=2.0, y=2.0, z=1.5)  # Default camera
    if correct_camera and "scene.camera" in correct_camera:
        camera_eye = correct_camera["scene.camera"]["eye"]
    fig_time.update_layout(
        title_x=0.5, title_y=0, title_yanchor='bottom',
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(x=0.7, len=0.7, title=None),
        scene=dict(yaxis=dict(autorange="reversed"), camera=dict(eye=camera_eye))
    )
    return fig_time


@app.callback(
    Output("line-r-correct", "figure"),
    Output("line-r-time", "figure"),
    Input("line-r-t-slider", "value"),
    Input("line-r-m-slider", "value"),
    Input("data-store", "data")
)
def update_line_r(t_val, m_val, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    df_r = df[(df["t"] == t_val) & (df["m"] == m_val)].sort_values("r")
    fig1 = px.line(df_r, x="r", y="correct_matches")
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig2 = px.line(df_r, x="r", y="time")
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig1.add_vline(x=default_r, line_dash="dash", line_color="red")
    fig2.add_vline(x=default_r, line_dash="dash", line_color="red")
    return fig1, fig2

@app.callback(
    Output("line-t-correct", "figure"),
    Output("line-t-time", "figure"),
    Input("line-t-r-slider", "value"),
    Input("line-t-m-slider", "value"),
    Input("data-store", "data")
)
def update_line_t(r_val, m_val, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    df_t = df[(df["r"] == r_val) & (df["m"] == m_val)].sort_values("t")
    fig1 = px.line(df_t, x="t", y="correct_matches")
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig2 = px.line(df_t, x="t", y="time")
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig1.add_vline(x=default_t, line_dash="dash", line_color="red")
    fig2.add_vline(x=default_t, line_dash="dash", line_color="red")
    return fig1, fig2

@app.callback(
    Output("line-m-correct", "figure"),
    Output("line-m-time", "figure"),
    Input("line-m-r-slider", "value"),
    Input("line-m-t-slider", "value"),
    Input("data-store", "data")
)
def update_line_m(r_val, t_val, data):
    df = pd.read_json(io.StringIO(data), orient='split')
    df_m = df[(df["r"] == r_val) & (df["t"] == t_val)].sort_values("m")
    fig1 = px.line(df_m, x="m", y="correct_matches")
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig2 = px.line(df_m, x="m", y="time")
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    fig1.add_vline(x=default_m, line_dash="dash", line_color="red")
    fig2.add_vline(x=default_m, line_dash="dash", line_color="red")
    return fig1, fig2

@app.callback(
    Output("r-slider", "value"),
    Output("t-slider", "value"),
    Output("m-slider", "value"),
    Output("line-r-t-slider", "value"),
    Output("line-r-m-slider", "value"),
    Output("line-t-r-slider", "value"),
    Output("line-t-m-slider", "value"),
    Output("line-m-r-slider", "value"),
    Output("line-m-t-slider", "value"),
    Input("reset-button", "n_clicks")
)
def reset_sliders(n_clicks):
    return default_r, default_t, default_m, default_t, default_m, default_r, default_m, default_r, default_t

@app.callback(
    Output("reset-feedback", "children"),
    Output("reset-feedback", "style"),
    Output("reset-feedback-timer", "disabled"),
    Output("reset-feedback-timer", "n_intervals"),
    Input("reset-button", "n_clicks"),
    Input("reset-feedback-timer", "n_intervals"),
    prevent_initial_call=True
)
def reset_feedback(n_clicks, n_intervals):
    base_style = {
        "position": "fixed",
        "top": "50%",
        "left": "50%",
        "transform": "translate(-50%, -50%)",
        "padding": "12px 20px",
        "backgroundColor": "#d4edda",
        "color": "#155724",
        "border": "1px solid #c3e6cb",
        "borderRadius": "8px",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.15)",
        "zIndex": 9999,
        "fontSize": "14px"
    }
    if ctx.triggered_id == "reset-button":
        return "✅ All sliders have been reset.", {**base_style, "display": "block"}, False, 0
    elif ctx.triggered_id == "reset-feedback-timer":
        return "", {**base_style, "display": "none"}, True, 0
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("r-slider", "min"), Output("r-slider", "max"), Output("r-slider", "marks"),
    Output("t-slider", "min"), Output("t-slider", "max"), Output("t-slider", "marks"),
    Output("m-slider", "min"), Output("m-slider", "max"), Output("m-slider", "marks"),
    Output("line-r-t-slider", "min"), Output("line-r-t-slider", "max"), Output("line-r-t-slider", "marks"),
    Output("line-r-m-slider", "min"), Output("line-r-m-slider", "max"), Output("line-r-m-slider", "marks"),
    Output("line-t-r-slider", "min"), Output("line-t-r-slider", "max"), Output("line-t-r-slider", "marks"),
    Output("line-t-m-slider", "min"), Output("line-t-m-slider", "max"), Output("line-t-m-slider", "marks"),
    Output("line-m-r-slider", "min"), Output("line-m-r-slider", "max"), Output("line-m-r-slider", "marks"),
    Output("line-m-t-slider", "min"), Output("line-m-t-slider", "max"), Output("line-m-t-slider", "marks"),
    Input("data-store", "data")
)
def update_slider_ranges(data_json):
    df = pd.read_json(io.StringIO(data_json), orient='split')
    r_vals = sorted(df["r"].unique())
    t_vals = sorted(df["t"].unique())
    m_vals = sorted(df["m"].unique())
    def marks(values):
        return {int(v): str(v) for v in values}
    return (
        min(r_vals), max(r_vals), marks(r_vals),
        min(t_vals), max(t_vals), marks(t_vals),
        min(m_vals), max(m_vals), marks(m_vals),
        min(t_vals), max(t_vals), marks(t_vals),
        min(m_vals), max(m_vals), marks(m_vals),
        min(r_vals), max(r_vals), marks(r_vals),
        min(m_vals), max(m_vals), marks(m_vals),
        min(r_vals), max(r_vals), marks(r_vals),
        min(t_vals), max(t_vals), marks(t_vals)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
