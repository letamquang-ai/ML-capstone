import gradio as gr
import models 

theme = gr.themes.Base(
    primary_hue="rose",
    secondary_hue="sky",
    neutral_hue="neutral",
    text_size="lg",
)

f =open("ui.html")
greeting= f.read()

f=open("background.html")
background=f.read()

f=open("component.css")
css=f.read()

model=models.model()

def show_selection(choice):
    return f"You selected: {choice}"
def return_accuracy(name):
    if name=='':
        return 0
    return (1-model.assess[name])*100

with gr.Blocks(
    theme=theme,
    css=css
               ) as demo: 
    with gr.Column(visible=True) as greeting_page:
        gr.Markdown(greeting,elem_classes='greeting')
        btn=gr.Button(value="Let'start!",elem_id='custom-button')

    with gr.Column(visible=False) as main_page:
        with gr.Tab("Price Display"):
            with gr.Row():
                plot=gr.LinePlot(x="date", y="cushing_crude_oil_price", title="Oil Price",y_title="price",scale=2,height=550)
                with gr.Column(scale=1):
                    year=gr.Dropdown(choices=["None"]+[str(i) for i in range(2003,model.now.year+1)],
                                    label="Year",
                                    value='None',
                                    )
                    month=gr.Dropdown(choices=["None","Jan","Feb",'Mar',"Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                                    label="Month",
                                    value='None'
                                    )
                    btn_plot=gr.Button("Search")    
                    gr.Markdown(
                        "### Note\n"
                        "- **Year: None**, **Month: None** → Displays the graph of price and oil \n"
                        "- **Year: None**, **Month: Selected** → Displays all days in the selected month of the most recent year.\n"
                        "- **Year: Selected**, **Month: None** → Displays all days in the selected year.",
                        elem_classes="note-box"
                    )
            btn_plot.click(fn=model.plot_for_price,inputs=[year,month],outputs=plot)
        with gr.Tab("Model Prediction"):
            with gr.Row():
                 plot_for_model=gr.LinePlot(x='Date', y='value', color='variable',
                    title='Prediction of cushing oil price', y_title='Price($)',
                    tooltip=['Date', 'variable', 'value'],scale=3,height=550)
                 with gr.Column(scale=1):
                    model_input=gr.Dropdown(choices=["Linear Regression","Support Vector Machine","Gaussian Process Regression","Ensemble Learning with Decision Trees"],
                                            label="Model",
                                            value="Linear Regression")
                    # Initial mode dropdown
                    mode_dropdown = gr.Dropdown(
                        choices=[""],
                        label="Mode",
                        interactive=True,
                        visible=False
                    )
                    def update_mode(model_name):
                        if model_name == "Support Vector Machine":
                            return gr.update(choices=["Linear", "Quadratic", "Cubic","Gaussian"], label="Kernel" ,value="Linear",visible=True,)
                        elif model_name == "Gaussian Process Regression":
                            return gr.update(choices=["Rational Quadratic", "Squared Exponential"], label="Kernel", value="Rational Quadratic",visible=True)
                        elif model_name == "Ensemble Learning with Decision Trees":
                            return gr.update(choices=["Bootstrap Aggregation", "Gradient Boosting"], label="Method", value="Bootstrap Aggregation",visible=True)
                        else:
                            return gr.update(choices=[""], value="",visible=False)
                    # Update mode dropdown based on model_input
                    model_input.change(fn=update_mode, inputs=model_input, outputs=mode_dropdown)
                    btn_model=gr.Button("Summit")    
            btn_model.click(fn=model.plot_model,inputs=[model_input,mode_dropdown],outputs=plot_for_model)  
        with gr.Tab("Predict Part"):
            with gr.Row():
                with gr.Column():
                    number_of_days=gr.Number(value=1,minimum=1,maximum=50,label="Days")
                    model_predict=gr.Dropdown(choices=["Linear Regression","Support Vector Machine","Gaussian Process Regression","Ensemble Learning"],label="Model")
                    btn_predict=gr.Button(value="Predict")
                    date=gr.DateTime(type='datetime',label='Date',include_time=False)
                    price=gr.Number(label="Price($)")
                    date.change(model.take_price,inputs=date,outputs=price)
                with gr.Column():
                    accuracy=gr.Number(label="Accuracy",interactive=False)
                    
                    bar_plot=gr.ScatterPlot(x="date",y='predict',y_title="Price($)",height=500)
            btn_predict.click(model.predict_for_future_days,inputs=[number_of_days,model_predict],outputs=[accuracy,bar_plot])

    def go_to():
        return gr.update(visible=False), gr.update(visible=True)
    
    btn.click(go_to,outputs=[greeting_page,main_page])
demo.launch()