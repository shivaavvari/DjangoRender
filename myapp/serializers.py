from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Mlmodel

class MlmodelSerializer(serializers.HyperlinkedModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')
    class Meta:
        model = Mlmodel
        fields = ['url','Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'owner']


class UserSerializer(serializers.ModelSerializer):
    mlmodels = serializers.HyperlinkedRelatedField(many=True, view_name='mlmodel-detail', read_only=True)

    class Meta:
        model = User
        fields = '__all__'