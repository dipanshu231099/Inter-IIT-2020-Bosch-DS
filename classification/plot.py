import os
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup


def plotgraphs(e,acu,val_acu,loss,val_loss,m,accuracy_score):
    soup = BeautifulSoup(open('graphs.html'), 'html.parser')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=acu, name='training accuracy',line=dict(color='firebrick'),mode='lines'))
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=val_acu, name='val accuracy',line=dict(color='royalblue'),mode='lines'))
    fig.update_layout(title='Accuracy', xaxis_title='Epochs',yaxis_title='accuracy')
    with open('/home/abhishek/django_project4/classification/templates/graphs.html', 'a') as f:
        soup.body.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=loss, name='training loss',line=dict(color='firebrick'),mode='lines'))
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=val_loss, name='val loss',line=dict(color='royalblue'),mode='lines'))
    fig.update_layout(title='Loss', xaxis_title='epochs',yaxis_title='loss')
    with open('/home/abhishek/django_project4/classification/templates/graphs.html', 'a') as f:
        soup.body.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))


    fig=px.imshow(m)

    fig.update_layout(title='accuracy = {}'.format(accuracy_score))
    with open('/home/abhishek/django_project4/classification/templates/graphs.html', 'a') as f:
        soup.body.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    with open("graphs.html", "w") as file:
    file.write(str(soup))
