import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'

const appRoot = fs.realpathSync(process.cwd())
const workspaceRoot = path.dirname(appRoot)
const workspaceName = path.basename(workspaceRoot)
const workspaceGroup = path.dirname(workspaceRoot)
const sharedDir = path.join(workspaceGroup, '.shared')
const sharedEnvPath = path.join(sharedDir, '.env.local')
const localEnvPath = path.join(appRoot, '.env.local')
const workspaceEnvPath = path.join(workspaceRoot, '.env.local')
const registryPath = path.join(sharedDir, 'workspace-ports.json')
const args = process.argv.slice(2)

function fileExists(filePath) {
  try {
    fs.accessSync(filePath, fs.constants.F_OK)
    return true
  } catch {
    return false
  }
}

function parseEnvFile(filePath) {
  const result = {}
  const lines = fs.readFileSync(filePath, 'utf8').split('\n')
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const eqIndex = trimmed.indexOf('=')
    if (eqIndex === -1) continue
    const key = trimmed.slice(0, eqIndex).trim()
    const val = trimmed.slice(eqIndex + 1).trim().replace(/^["']|["']$/g, '')
    result[key] = val
  }
  return result
}

function patchSecretsFromHome() {
  const secretsPath = path.join(os.homedir(), '.copilot-hackathon.env')
  if (!fileExists(secretsPath)) return
  if (!fileExists(sharedEnvPath)) return

  const secrets = parseEnvFile(secretsPath)
  let sharedContent = fs.readFileSync(sharedEnvPath, 'utf8')
  let changed = false

  for (const [key, val] of Object.entries(secrets)) {
    if (!val) continue
    const emptyPattern = new RegExp(`^(${key}=["']?["']?)$`, 'm')
    if (emptyPattern.test(sharedContent)) {
      sharedContent = sharedContent.replace(emptyPattern, `${key}="${val}"`)
      changed = true
    }
  }

  if (changed) {
    const target = fs.lstatSync(sharedEnvPath).isSymbolicLink()
      ? fs.realpathSync(sharedEnvPath)
      : sharedEnvPath
    fs.writeFileSync(target, sharedContent)
    console.log(`[workspace-dev] Patched secrets from ${secretsPath}`)
  }
}

function ensureSharedEnv() {
  fs.mkdirSync(sharedDir, { recursive: true })

  if (!fileExists(sharedEnvPath) && fileExists(localEnvPath)) {
    fs.copyFileSync(localEnvPath, sharedEnvPath)
  }

  if (!fileExists(sharedEnvPath) && fileExists(workspaceEnvPath)) {
    fs.copyFileSync(workspaceEnvPath, sharedEnvPath)
  }

  if (!fileExists(sharedEnvPath)) {
    console.warn(
      `[workspace-dev] No shared env found at ${sharedEnvPath}. Auth may fail until it exists.`
    )
    return
  }

  if (!fileExists(localEnvPath)) {
    fs.symlinkSync(sharedEnvPath, localEnvPath)
    return
  }

  const stat = fs.lstatSync(localEnvPath)
  if (stat.isSymbolicLink()) {
    const target = path.resolve(path.dirname(localEnvPath), fs.readlinkSync(localEnvPath))
    if (target !== sharedEnvPath) {
      console.warn(
        `[workspace-dev] ${localEnvPath} points to ${target}, not the shared env. Leaving it unchanged.`
      )
    }
    return
  }

  const localContents = fs.readFileSync(localEnvPath, 'utf8')
  const sharedContents = fs.readFileSync(sharedEnvPath, 'utf8')

  if (localContents !== sharedContents) {
    console.warn(
      `[workspace-dev] ${localEnvPath} differs from the shared env. Leaving it unchanged.`
    )
  }
}

function loadRegistry() {
  if (!fileExists(registryPath)) return {}

  try {
    return JSON.parse(fs.readFileSync(registryPath, 'utf8'))
  } catch {
    console.warn(
      `[workspace-dev] Could not parse ${registryPath}. Recreating the port registry.`
    )
    return {}
  }
}

function saveRegistry(registry) {
  fs.mkdirSync(sharedDir, { recursive: true })
  fs.writeFileSync(registryPath, `${JSON.stringify(registry, null, 2)}\n`)
}

function detectExplicitPort(cliArgs) {
  for (let index = 0; index < cliArgs.length; index += 1) {
    const arg = cliArgs[index]
    if (arg === '--port' || arg === '-p') {
      const value = cliArgs[index + 1]
      return value ? Number.parseInt(value, 10) : null
    }
    if (arg.startsWith('--port=')) {
      return Number.parseInt(arg.split('=')[1], 10)
    }
  }

  return null
}

function canListen(port) {
  return new Promise((resolve) => {
    const server = net.createServer()

    server.once('error', () => {
      resolve(false)
    })

    server.once('listening', () => {
      server.close(() => resolve(true))
    })

    server.listen(port, '127.0.0.1')
  })
}

async function assignPort() {
  const explicitPort = detectExplicitPort(args)
  if (explicitPort) return explicitPort

  const registry = loadRegistry()
  const assignedPort = registry[workspaceName]

  if (assignedPort) {
    const available = await canListen(assignedPort)
    if (!available) {
      console.error(
        `[workspace-dev] Port ${assignedPort} is already in use for workspace "${workspaceName}". Stop the existing server or run with --port to override.`
      )
      process.exit(1)
    }
    return assignedPort
  }

  const reservedPorts = new Set(
    Object.values(registry).filter((value) => Number.isInteger(value))
  )

  let candidate = 3001
  while (reservedPorts.has(candidate) || !(await canListen(candidate))) {
    candidate += 1
  }

  registry[workspaceName] = candidate
  saveRegistry(registry)

  return candidate
}

function startNextDev(port) {
  const nextBin = path.join(appRoot, 'node_modules', '.bin', 'next')
  const nextArgs = ['dev', '--port', String(port)]

  if (detectExplicitPort(args)) {
    const filteredArgs = []
    for (let index = 0; index < args.length; index += 1) {
      const arg = args[index]
      if (arg === '--port' || arg === '-p') {
        index += 1
        continue
      }
      if (arg.startsWith('--port=')) continue
      filteredArgs.push(arg)
    }
    nextArgs.push(...filteredArgs)
  } else {
    nextArgs.push(...args)
  }

  console.log(`[workspace-dev] Workspace: ${workspaceName}`)
  console.log(`[workspace-dev] URL: http://localhost:${port}`)
  if (fileExists(sharedEnvPath)) {
    console.log(`[workspace-dev] Shared env: ${sharedEnvPath}`)
  }

  const child = spawn(nextBin, nextArgs, {
    cwd: appRoot,
    stdio: 'inherit',
    shell: false,
  })

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal)
      return
    }
    process.exit(code ?? 0)
  })
}

ensureSharedEnv()
patchSecretsFromHome()
const port = await assignPort()
startNextDev(port)
