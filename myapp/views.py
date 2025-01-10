
"""
This module contains views for the Django application, including API endpoints and viewsets for handling Mlmodel and User objects.
It also includes functions for training a machine learning model, making predictions, and rendering forms.
Classes:
    MlmodelViewSet: A viewset for viewing and editing Mlmodel instances.
    UserViewSet: A viewset for viewing and editing User instances.
Functions:
    api_root(request, format=None): API root endpoint providing links to user and mlmodel lists.
    get_mlmodel(request, format=None): Handles GET and POST requests for the Mlmodel form.
    predict(request, format=None): Handles POST requests to make predictions using the latest Mlmodel instance.
    Train(request): Handles POST requests to train a machine learning model using a CSV file.
    thanks(request, format=None): Renders a thank you page after data is saved to the model.
"""
from rest_framework.response import Response
from .permissions import IsOwnerOrReadOnly
from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import permissions
from .serializers import MlmodelSerializer ,UserSerializer
from .models import Mlmodel
from rest_framework.decorators import api_view ,permission_classes
from rest_framework.reverse import reverse
from rest_framework import renderers
from rest_framework.decorators import action
from django.contrib.auth.models import User
from rest_framework.renderers import StaticHTMLRenderer
from django.http import HttpResponseRedirect, HttpResponse
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import roc_auc_score ,roc_curve, f1_score,precision_score,recall_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from .forms import MlmodelForm
from rest_framework.permissions import IsAuthenticated
# Create  your views here.

@api_view(['GET', 'POST'])
@permission_classes(( ))
def api_root(request, format=None):
    
    try:
         return Response({
        'users':    reverse('user-list', request=request, format=format),
        'mlmodels': reverse('mlmodel-list', request=request, format=format),
          })      
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    




class MlmodelViewSet(viewsets.ModelViewSet):
   
    queryset = Mlmodel.objects.all()
    serializer_class = MlmodelSerializer
    permission_classes = [permissions.IsAuthenticated,
                          IsOwnerOrReadOnly]  
    
    @action(detail=True, methods=['post','get'],renderer_classes=[renderers.StaticHTMLRenderer])  
    def perform_predict(self, request ,pk,**kwargs):
        try:
           
            mlmodel = Mlmodel.objects.get(pk=pk)
            serializer = MlmodelSerializer(mlmodel, context={'request': self.request})
            data = serializer.data
            data_list = list(data.values())
            data_list = data_list[1:-1]    
            data_list = np.array(data_list).reshape(1,-1)
            with open("model_rf.pkl","rb") as f:
                model = pickle.load(f)
                f.close()
            predict_class = model.predict(data_list)    
            context = {
            "data": predict_class,
              }              

            return render(self.request, 'predict.html', context=context)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
    @action(detail=False, methods=['post','get'],renderer_classes=[renderers.StaticHTMLRenderer])  
    def navigate_form(self, request ,**kwargs):
        try:

            return HttpResponseRedirect('/thanks')
        except Exception as e:
            return Response({"error": str(e)}, status=500)

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer




@api_view(['GET', 'POST'])
@permission_classes(())
def get_mlmodel(request, format=None):
    if request.method == 'POST':
        form = MlmodelForm(request.POST)
        

        
        if form.is_valid():
            form.save()
        queryset = Mlmodel.objects.last()
        return HttpResponseRedirect('/thanks/')
        
    else:
        form = MlmodelForm()
        context = {
            "form": form,
            'users': reverse('user-list', request=request, format=format),
            'mlmodels': reverse('mlmodel-list', request=request, format=format),
        }              
        return render(request, 'mlmodel.html', context=context)


@api_view(['GET', 'POST'])
@permission_classes(())
def predict(request,format=None):
  try:
    
     
        mlmodel = Mlmodel.objects.last()
        serializer = MlmodelSerializer(mlmodel, context={'request': request})
        data = serializer.data
        data_list = list(data.values())
        data_list = data_list[1:-1]    
        data_list = np.array(data_list).reshape(1, -1)
        with open("model_rf.pkl", "rb") as f:
            model = pickle.load(f)
            f.close()
        predict_class = model.predict(data_list) 
        context = {
            "data": predict_class,
            'users': reverse('user-list', request=request, format=format),
            'mlmodels': reverse('mlmodel-list', request=request, format=format),
        }              
        return render(request, 'predict.html', context=context)
  except Exception as e:
        return Response({"error": str(e)}, status=500)





@api_view(['GET', 'POST'])
@permission_classes(())
def train(request):
 try:
    if request.method =="POST":   
       df = pd.read_csv("creditcard.csv")
       X =df[df.columns[:-1]][0:10000]
       y = df[df.columns[-1]][0:10000]
       scaler  = StandardScaler()
       scaler.fit_transform(X)
       X_scaled = scaler.transform(X)
       X_train,X_test,y_train,y_test = train_test_split(X_scaled, y,test_size=0.33,random_state=42)
     # cls = RandomForestClassifier() 
       with open("model_rf.pkl","rb") as f:
          cls = pickle.load(f)
          f.close() 
       cls.fit(X_train,y_train )
       y_pred = cls.predict(X_test)
       classificationreport = classification_report(y_test,y_pred)
       f1score = f1_score(y_test,y_pred)
       precisionscore =precision_score(y_test,y_pred)
       recallscore = recall_score(y_test,y_pred)
       accuracyscore = accuracy_score(y_test,y_pred)
       y_score = cls.predict_proba(X_test)
       y_score =y_score[:,1]
       rocaucscore= roc_auc_score(y_test,y_score)
      

       return(render(request,'train.html',{ "data": "Data trained",
                                            'classificationreport':classificationreport,
                                             'f1score':f1score,
                                              'precisionscore':precisionscore,
                                               'recallscore':recallscore,
                                               'accuracyscore':accuracyscore,
                                               'rocaucscore':rocaucscore,                                                             
                                            }))
    else:
        
        return(render(request,'train.html',{"data":"Data ready for training ",
                                            }))
 except Exception as e:
    return Response({"error": str(e)}, status=500)

@api_view(['GET', 'POST'])
@permission_classes(())
def thanks(request, format=None):
    
        try:
            return render(request, "mlmodel.html", {
                "data": "The data is saved to the model.",
                'users': reverse('user-list', request=request, format=format),
                'mlmodels': reverse('mlmodel-list', request=request, format=format),
            })
        except Exception as e:
            return Response({"error": str(e)}, status=500)
