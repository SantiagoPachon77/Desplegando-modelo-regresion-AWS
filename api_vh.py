#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m1_model_deployment import predict_price_vh

# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Price Vehiculos Prediction API',
    description='Price Vehiculos Prediction API')

ns = api.namespace('predict', 
     description='Price Vehiculos Predict')

# Definición argumentos o parámetros de la API
parser = api.parser()

parser.add_argument(
    'Year',
    type = int,
    required = True)

parser.add_argument(
    'Mileage',
    type = int,
    required = True)

parser.add_argument(
    'State',
    type = str,
    required = True
)

parser.add_argument(
    'Make',
    type = str,
    required = True
)

parser.add_argument(
    'Model',
    type = str,
    required = True
    )

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_price_vh(args['Year'], args['Mileage'], args['State'], args['Make'], args['Model'])
        }, 200
    
    
if __name__ == '__main__':
    # Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
