using Microsoft.ML;
using SentimentAnalysisML.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SentimentAnalysis.Services
{
    public static class Sentiment
    {
        //Load the model
        static MLContext context = new MLContext();
        static DataViewSchema modelSchema; 
                                                        //need to change path 
        static ITransformer model = context.Model.Load(@"C:\Users\zheny\source\repos\SentimentAnalysis\SentimentAnalysisML.Model\MLModel.zip", out modelSchema);

        [ThreadStatic]
        static PredictionEngine<ModelInput, ModelOutput> t_engine;
        public static PredictionEngine<ModelInput,ModelOutput> GetPredictionEngine()
        {
            if (t_engine != null)
                return t_engine;

            return t_engine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
        }

        public static ModelOutput Predict(string text)
        {
            var predEngine = GetPredictionEngine();

            return predEngine.Predict(new ModelInput { Review = text });
        }

        public static int EvaluatePrediction(string text)
        {
            if (text != "")
            {
                ModelOutput prediction = Predict(text);

                switch (prediction.Score[0] > prediction.Score[1])
                {
                    case true:
                        return 1;
                    case false:
                        return -1;
                    default:
                        return 0;
                }

            }
            
            return 0;
        }
    }
}
