import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import webbrowser
from threading import Timer
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Load the data
file_path = 'umap_df.tsv'  # Update this path if needed
umap_df = pd.read_csv(file_path, sep='\t')
umap_df['id'] = range(len(umap_df))
umap_df['strand'] = umap_df['strand'].fillna('unknown')
overall_type_proportions = umap_df['cCRE_type'].value_counts(normalize=True)

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout using Bootstrap components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("UMAP Visualization with Lasso Selection", className="text-center my-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Label("Color by:"),
            dcc.Dropdown(
                id='color-option',
                options=[
                    {'label': 'cCRE Type', 'value': 'cCRE_type'},
                    {'label': 'Strand', 'value': 'strand'}
                ],
                value='cCRE_type'
            )
        ], width=3, className="mb-4")
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='umap-scatter',
                figure={},
                config={'modeBarButtonsToAdd': ['lasso2d']},
                style={'height': '100%'}
            )
        ], className="mb-4", width=11),
        
        dbc.Col([
            dbc.Label("Opacity:"),
            dcc.Slider(
                id='opacity-slider',
                min=0,
                max=1,
                step=0.05,
                value=0.5,  
                vertical=True,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], className="mb-4")
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("Selection Info", className="mb-0")),
                dbc.CardBody(id='selection-info')
            ])
        ], className="mb-4")
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Download Selection", 
                id="btn-download", 
                color="primary", 
                className="me-2",
                style={"display": "none"}, 
            ),
            dcc.Download(id="download-selection")
        ], width={"size": 3}),
    ], className="mb-4")
], fluid=True)

@app.callback(
    Output("btn-download", "style"),
    Input("umap-scatter", "selectedData")
)
def toggle_button_visibility(selectedData):
    if selectedData:
        return {"display": "block"}
    else:
        return {"display": "none"}

@app.callback(
    Output('umap-scatter', 'figure'),
    [Input('color-option', 'value'), Input('opacity-slider', 'value')]
)
def update_scatter(color_by, opacity_value):
    if color_by == 'cCRE_type':
        fig = px.scatter(
            umap_df, x='UMAP1', y='UMAP2', color=color_by,
            color_discrete_map={'dELS': '#FFCD00', 'PLS': '#FF0000'},
            opacity=opacity_value,
            custom_data=['id'],
            hover_data=['cCRE_type', 'chrom', 'start', 'end', 'rDHS'],
            labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2'},
            height=800
        )
    elif color_by == 'strand':
        fig = px.scatter(
            umap_df, x='UMAP1', y='UMAP2', color=color_by,
            color_discrete_map={'+': '#0066CC', '-': '#CC0000', 'Bidirectional': '#9933CC', 'unknown': 'gray'},
            opacity=opacity_value,
            custom_data=['id'],
            hover_data=['cCRE_type', 'chrom', 'start', 'end', 'rDHS', 'strand'],
            labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2'},
            height=800
        )
    
    fig.update_layout(
        dragmode='lasso',
        clickmode='event+select',
        template="plotly_white",
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig

# Callback to display selection info
@app.callback(
    Output('selection-info', 'children'),
    [Input('umap-scatter', 'selectedData')]
)
def display_selected_data(selectedData):
    if not selectedData or not selectedData.get('points'):
        return "No points selected"
    
    selected_ids = [p['customdata'][0] for p in selectedData['points']]
    selected_df = umap_df[umap_df['id'].isin(selected_ids)]
    num_selected = len(selected_df)
    
    type_counts = selected_df['cCRE_type'].value_counts().reset_index()
    type_counts.columns = ['cCRE Type', 'Count']

    type_counts['Selection %'] = (type_counts['Count'] / num_selected) * 100
    type_counts['Overall %'] = type_counts['cCRE Type'].map(overall_type_proportions) * 100

    type_counts['Enrichment Factor'] = type_counts.apply(
        lambda row: (row['Selection %'] / row['Overall %']) if row['Overall %'] > 0 else float('inf'),
        axis=1
    )
    
    # Get chromosome distribution
    chrom_counts = selected_df['chrom'].value_counts().reset_index()
    chrom_counts.columns = ['Chromosome', 'Count']
    
    # Calculate average length
    avg_length = selected_df['length'].mean()

    if len(selected_df) > 5:  # Only create violin plot if there are enough data points
        violin_fig = px.violin(
            selected_df, 
            x='cCRE_type', 
            y='length', 
            color='cCRE_type',
            color_discrete_map={'dELS': '#FFCD00', 'PLS': '#FF0000'},
            box=True,
            points=False,
            template="plotly_white"
        )
        violin_fig.update_layout(
            xaxis_title="cCRE Type",
            yaxis_title="Length (bp)",
            showlegend=False,
            height=800
        )
        violin_plot = dcc.Graph(figure=violin_fig)
    else:
        violin_plot = html.P("Not enough data points selected for a meaningful violin plot")

    
    # Create summary with Bootstrap components
    summary_info = [
        dbc.Alert(f"Selected {len(selected_df)} points", color="info", className="mb-3"),
        dbc.Badge(f"Average length: {avg_length:.2f} bp", color="secondary", className="mb-3 p-2"),

        html.H5("cCRE Type Distribution & Enrichment", className="mt-4"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Type"),
                html.Th("Count"),
                html.Th("Selection %"), # Renamed/Added
                html.Th("Overall %"),   # Added
                html.Th("Enrichment")   # Added (Factor)
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row['cCRE Type']),
                    html.Td(row['Count']),
                    html.Td(f"{row['Selection %']:.2f}%"), # Format percentage
                    html.Td(f"{row['Overall %']:.2f}%"),   # Format percentage
                    html.Td('inf' if np.isinf(row['Enrichment Factor']) else f"{row['Enrichment Factor']:.2f}")
                ])
                for _, row in type_counts.iterrows()
            ])
        ], striped=True, bordered=True, hover=True, className="mb-4"),
        

        html.H5("Length Distribution by cCRE Type", className="mt-4"),
        violin_plot,
        
        html.H5("Chromosome Distribution", className="mt-4"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Chromosome"), 
                html.Th("Count")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row['Chromosome']), 
                    html.Td(row['Count'])
                ])
                for _, row in chrom_counts.head(10).iterrows()
            ])
        ], striped=True, bordered=True, hover=True)
    ]
    
    return summary_info

# Callback for downloading selected data
@app.callback(
    Output("download-selection", "data"),
    [Input("btn-download", "n_clicks")],
    [State('umap-scatter', 'selectedData')]
)
def download_selection(n_clicks, selectedData):
    if n_clicks is None or not selectedData or not selectedData.get('points'):
        raise PreventUpdate
    
    selected_ids = [p['customdata'][0] for p in selectedData['points']]
    selected_df = umap_df[umap_df['id'].isin(selected_ids)]
    
    # Return a CSV of the selected data
    return dcc.send_data_frame(selected_df.to_csv, "selected_points.tsv", sep='\t', index=False)

# def open_browser():
#     webbrowser.open_new("http://localhost:8050")

# Run the app
if __name__ == '__main__':
    # Timer(1.5, open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=8050)