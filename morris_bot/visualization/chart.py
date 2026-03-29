import os
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from morris_bot.indicators.base import BaseIndicator, NoIndicator


class ChartVisualizer:

    @staticmethod
    def create_screenshot(df: pd.DataFrame,
                          ticker: str,
                          tf: str,
                          pattern: str,
                          signal_time: datetime,
                          indicator: BaseIndicator,
                          output_dir: str = "/tmp") -> Optional[str]:
        try:
            matches = df.index[df['datetime'] == signal_time].tolist()
            if not matches:
                return None
            idx     = matches[0]
            plot_df = df.iloc[max(0, idx - 30):min(len(df), idx + 5)].copy()

            ind_col = indicator.column_name
            has_ind = ind_col in plot_df.columns and not isinstance(indicator, NoIndicator)

            rows        = 2 if has_ind else 1
            row_heights = [0.7, 0.3] if has_ind else [1.0]

            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights,
                subplot_titles=[f"{ticker} {tf}", indicator.plot_label if has_ind else ""]
            )

            fig.add_trace(go.Candlestick(
                x=plot_df['datetime'],
                open=plot_df['open'], high=plot_df['high'],
                low=plot_df['low'],   close=plot_df['close'],
                name=ticker,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df['datetime'], y=plot_df['ema10'],
                line=dict(color='orange', width=1.5),
                name='EMA10'
            ), row=1, col=1)

            is_bullish = any(x in pattern.lower() for x in
                             ['bull', 'hammer', 'morning', 'soldier', 'piercing'])
            color  = "#00e676" if is_bullish else "#ff1744"
            y_val  = plot_df.loc[idx, 'low'] if is_bullish else plot_df.loc[idx, 'high']
            ay_val = -40 if is_bullish else 40

            fig.add_annotation(
                x=signal_time, y=y_val,
                text=f"<b>{pattern}</b>",
                showarrow=True, arrowhead=2,
                arrowcolor=color, bgcolor=color,
                font=dict(color="black", size=10),
                ay=ay_val, row=1, col=1
            )

            if has_ind:
                ind_values = plot_df[ind_col]

                if "macd" in ind_col:
                    bar_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in ind_values]
                    fig.add_trace(go.Bar(
                        x=plot_df['datetime'], y=ind_values,
                        marker_color=bar_colors, name=indicator.plot_label
                    ), row=2, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=plot_df['datetime'], y=ind_values,
                        line=dict(color='#7c4dff', width=1.5),
                        name=indicator.plot_label
                    ), row=2, col=1)

                for level in indicator.get_level_lines():
                    fig.add_hline(
                        y=level["value"],
                        line=dict(color=level["color"], dash=level["dash"], width=1),
                        row=2, col=1
                    )

            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                title=dict(text=f"{ticker} | {tf} | {pattern}", font=dict(size=14)),
                height=600,
                showlegend=False,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='#2a2a2a')

            safe_pattern = pattern.replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(output_dir, f"alert_{ticker}_{tf}_{safe_pattern}.png")
            fig.write_image(path, scale=2)
            return path

        except Exception as e:
            print(f"[ChartVisualizer] Ошибка: {e}")
            return None
