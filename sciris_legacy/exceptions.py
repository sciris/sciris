"""
exceptions.py -- code for custom werkzeug exceptions
    
Last update: 12/14/17 (gchadder3)
"""

from werkzeug.exceptions import HTTPException


class BaseRESTException(HTTPException):
    code = 500
    _message = 'An unexpected error happened'

    def __init__(self):
        self.description = self._message
        super(BaseRESTException, self).__init__()

    def to_dict(self):
        return {
            'status_code': self.code,
            'userMessage': self.description,
            'internalMessage': self.__str__()
        }
        
        
class RecordDoesNotExist(BaseRESTException):
    code = 410
    _message = 'The resource you are looking for does not exist'
    _model = 'Resource'

    def __init__(self, id=None, model=None, project_id=None):
        super(RecordDoesNotExist, self).__init__()
        if id is not None or model != 'Resource':
            elements = [
                'The {}'.format(model if model is not None else self._model),
                'with id {}'.format(id) if id is not None else 'you are looking for',
                'in project {}'.format(project_id) if project_id else '',
                'does not exist'
            ]
            self.description = ' '.join(elements)


class ProjectDoesNotExist(RecordDoesNotExist):
    _model = 'project' 


class SpreadsheetDoesNotExist(RecordDoesNotExist):
    _model = 'spreadsheet'
    