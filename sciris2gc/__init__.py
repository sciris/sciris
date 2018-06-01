# Specify the version, for the purposes of figuring out which version was used to create a project
from .version import version, versiondate

# Print the license
sciris_license = 'Sciris %s (%s)' % (version, versiondate)
print(sciris_license)