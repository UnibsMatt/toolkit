import bokeh.io
from bokeh.models import BoxSelectTool, ColumnDataSource
from bokeh.palettes import Category10_10
from bokeh.plotting import figure, show
from bokeh.transform import factor_mark


def awesome_plot(dataframe):
    tooltips = """
        <div>
            <div>
                <img
                    src="" height="150" alt="@imgs" width="450"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>

            </div>
            <div>
                <img src="@mask" height="150" alt="@mask" width="450"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"></img>
            </div>
            <div>
                <span>Classe: @class_prod</span>
            </div>
            <div>
                <span>Prodotto: @prod</span>
            </div>
            <div>
                <span style="font-size: 15px;">Posizione</span>
                <span style="font-size: 10px; color: #696;">(@x{1.11}, @y{1.11})</span>
            </div>
        </div>
    """
    bokeh.io.output_notebook()
    markers = ["hex", "triangle", "circle_x"]

    resurces = []

    p = figure(title="Distribuzione", plot_width=1000, plot_height=1000)
    p.add_tools(BoxSelectTool())

    p.xaxis.axis_label = "x"
    p.yaxis.axis_label = "y"
    for cl, color, mark in zip(sorted(dataframe["class_prod"].unique()), Category10_10, markers):
        src = ColumnDataSource(dataframe.loc[dataframe["class_prod"] == cl])
        p.scatter(
            "x",
            "y",
            line_alpha="line_alpha",
            fill_alpha="fill_alpha",
            size="size",
            legend_label=str(cl),
            color=color,
            marker=factor_mark("prod", markers, dataframe["prod"].unique()),
            muted_alpha=0.2,
            source=src,
        )
        resurces.append(src)
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    show(p)
