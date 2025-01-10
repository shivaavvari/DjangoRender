from rest_framework import routers
from django.urls import path, include
from myapp import views
from rest_framework import permissions
from drf_yasg.views import  get_schema_view
from drf_yasg import openapi
router = routers.DefaultRouter()
router.register(r'mlmodels',views.MlmodelViewSet,  basename='mlmodel')
router.register(r'users',views.UserViewSet,  basename='user')

schema_view = get_schema_view(
   openapi.Info(
      title="Creditcard Fraud Detection Model",
      default_version='1.0',
      description="Check the fraud in Credit card Transactions",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="aisshiva766@gmail.com"),
      license=openapi.License(name="BSD License"),
      
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)



urlpatterns =[

   path('swagger<format>/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
   path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
   path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
   path('', include(router.urls)),
   path('api_route/',views.api_root),
   path('form/', views.get_mlmodel),
   path('predict/', views.predict),
   path('predict/<int:id>', views.predict),
   path('train/', views.train),
   path('thanks/', views.thanks),
    
    
    
]
