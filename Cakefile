{print}       = require 'util'
{spawn, exec} = require 'child_process'
watchr        = require 'watchr'
path          = require 'path'
wrench        = require 'wrench'
colors        = require 'colors'

srcDir = '.'
doccoFlags = ' -t ~/dev/assets/templates/docco/parallel-latex/docco.jst'

blue = '\x1b[34m'
colors.setTheme
  verbose  : 'black'
  debug    : 'blue'
  error    : 'red'
  warn     : 'yellow'
  info     : 'green'
  emph     : 'inverse'
  underline: 'underline'
  data     : 'blue'

log = (message, styles, callback) ->
  if styles?
    for style in styles
      message = message[style]
  console.log message

  callback?()
execute = (cmd, options, callback) ->
  command = spawn cmd, options
  command.stdout.pipe process.stdout
  command.stderr.pipe process.stderr
  command.on 'exit', (status) ->
    callback?() if status is 0

docco = (callback) ->
  files = wrench.readdirSyncRecursive(srcDir)
  files = (path.join(srcDir,file) for file in files when /\.py$/.test file)
  cmd = 'docco' + doccoFlags
  for file in files
    cmd = cmd + ' ' +file
  console.log blue
  console.log cmd
  exec cmd, (err, stdout, stderr) ->
    log "└── Generated project documentation \n", ['info']
    callback?()

watch = ->
  watchr.watch {
    path: srcDir
  , listeners: {
      log:
        (logLevel) ->
          # console.log 'watchr log: '.data, arguments
    , error:
        (err) -> log "watchr error: #{err}", ['error']
    , watching:
        (err, watcherInstance, isWatching) ->
          if (err)
            log "Failed to watch #{watcherInstance.path} with error:", ['error']
            log err, ['error']
          else
            log "\n Watching files in #{watcherInstance.path} ", ['data', 'emph']
    , change:
        (changeType, filePath, fileCurrentStat, filePreviousStat) ->
          if changeType is 'create'
            switch path.extname(filePath)
              when '.py'
                invoke 'docs'
    }
  }

task 'docs', 'Generate annotated source code with Docco', ->
  docco()

task 'watch', 'Watch for changes and generate annoted source code with Docco', ->
  docco()
  watch()
