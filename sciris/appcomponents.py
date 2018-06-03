"""
appcomponents.py -- classes for Sciris app components
    
Last update: 5/7/18 (gchadder3)
"""

import mpld3 # For plotting

class AppComponent(object):
    def __init__(self, content=None):
        self.content = content
    
    def render(self):
        pass
    
class HtmlComponent(AppComponent):
    def __init__(self, content=None):
        super(HtmlComponent, self).__init__(content)
    
    def render(self):
        pass
    
class CoreComponent(AppComponent):
    def __init__(self, content=None):
        super(CoreComponent, self).__init__(content)
    
    def render(self):
        pass
    
 
class H1(HtmlComponent):
    def __init__(self, content=''):
        super(H1, self).__init__(content)
        
    def render(self):
        return '<h1>%s</h1>' % self.content
        
class P(HtmlComponent):
    def __init__(self, content=''):
        super(P, self).__init__(content)
        
    def render(self):
        return '<p>%s</p>' % self.content 
    
class Graph(CoreComponent):
    def __init__(self, content=None):
        super(Graph, self).__init__(content)
        
    def render(self):
        return mpld3.fig_to_html(self.content) 