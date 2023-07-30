# labeling config

# <View>
#   <View style="display:flex;align-items:start;gap:8px;flex-direction:row"><Image name="image" value="$image" zoomControl="true"/><View><Filter toName="label" minlength="0" name="filter"/><RectangleLabels name="label" toName="image" showInline="false">


#   <Label value="text" background="#b3b3b3"/><Label value="title" background="#6b6b6b"/><Label value="figure" background="#ffe8c7"/><Label value="figure_caption" background="#ffb752"/><Label value="table" background="#D3F261"/><Label value="table_caption" background="#389E0D"/><Label value="header" background="#70fff5"/><Label value="footer" background="#2e7a79"/><Label value="reference" background="#ADC6FF"/><Label value="equation" background="#9254DE"/><Label value="caption_block" background="#ff0d00"/><Label value="divider" background="#00ff04"/></RectangleLabels></View></View>

# </View>

from label_studio_ml.model import LabelStudioMLBase


class CaseSegModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to call base class constructor
        super(CaseSegModel, self).__init__(**kwargs)

        # you can preinitialize variables with keys needed to extract info from tasks and annotations and form predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns
            the list of predictions based on input list of tasks
        """
        predictions = []
        for task in tasks:
            predictions.append({
                'score': 0.987,  # prediction overall score, visible in the data manager columns
                'model_version': 'delorean-20151021',  # all predictions will be differentiated by model version
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'score': 0.5,  # per-region score, visible in the editor
                    'value': {
                        'choices': [self.labels[0]]
                    }
                }]
            })
        return predictions

    def fit(self, annotations, **kwargs):
        """ This is where training happens: train your model given list of annotations,
            then returns dict with created links and resources
        """
        return {'path/to/created/model': 'my/model.bin'}