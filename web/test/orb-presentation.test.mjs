import assert from 'node:assert/strict'
import test from 'node:test'
import {
  desktopOrbClassName,
  resolveOrbActivity,
  resolveOrbVisualState,
} from '../src/desktop/orb-presentation.js'

test('keeps speaking animation active while microphone input is muted', () => {
  assert.equal(desktopOrbClassName({
    state: 'speaking',
    enabled: false,
  }), 'desktop-orb-stage speaking input-muted orb-moving orb-lit')
})

test('exposes semantic listening, error, and dragging states to the desktop orb', () => {
  assert.equal(desktopOrbClassName({
    state: 'listening',
    enabled: true,
    error: true,
    dragging: true,
  }), 'desktop-orb-stage listening enabled error dragging orb-moving orb-lit')
})

test('keeps the waking lifecycle distinct from an error', () => {
  const className = desktopOrbClassName({
    state: 'waking',
    enabled: true,
    lifecycle: 'waking',
  })
  assert.match(className, /\bwaking\b/)
  assert.doesNotMatch(className, /\berror\b/)
})

test('arbitrates visual states by layered priority', () => {
  // 满载输入：每一层压过下一层。
  const everything = {
    lifecycle: 'sleeping',
    runtimeState: 'starting',
    connectionError: true,
    connecting: true,
    ownershipBusy: true,
    voiceState: 'listening',
    tasksWorking: true,
  }
  assert.equal(resolveOrbVisualState(everything), 'sleeping')
  assert.equal(
    resolveOrbVisualState({ ...everything, lifecycle: 'waking' }),
    'waking',
  )
  assert.equal(
    resolveOrbVisualState({ ...everything, lifecycle: 'active' }),
    'starting',
  )
  assert.equal(
    resolveOrbVisualState({
      ...everything,
      lifecycle: 'active',
      runtimeState: 'failed',
    }),
    'error',
  )
  // 对话态优先于所有后台态。
  for (const voiceState of ['listening', 'processing', 'speaking']) {
    assert.equal(resolveOrbVisualState({
      ...everything,
      lifecycle: 'active',
      runtimeState: 'ready',
      connectionError: false,
      voiceState,
    }), voiceState)
  }
  // 后台态优先级：working > occupied > connecting > idle。
  const background = {
    lifecycle: 'active',
    runtimeState: 'ready',
    connectionError: false,
    connecting: true,
    ownershipBusy: true,
    voiceState: 'idle',
    tasksWorking: true,
  }
  assert.equal(resolveOrbVisualState(background), 'working')
  assert.equal(resolveOrbVisualState({
    ...background,
    tasksWorking: false,
  }), 'occupied')
  assert.equal(resolveOrbVisualState({
    ...background,
    tasksWorking: false,
    ownershipBusy: false,
  }), 'connecting')
  assert.equal(resolveOrbVisualState({
    ...background,
    tasksWorking: false,
    ownershipBusy: false,
    connecting: false,
  }), 'idle')
  // 无入参时待机；未知 Provider 状态不能泄漏给皮肤。
  assert.equal(resolveOrbVisualState(), 'idle')
  assert.equal(resolveOrbVisualState({ voiceState: 'custom' }), 'idle')
})

test('waits still and dim until recording starts or the wake word activates the orb', () => {
  assert.deepEqual(resolveOrbActivity(), { moving: false, illuminated: false })
  assert.deepEqual(resolveOrbActivity({ lifecycle: 'sleeping', ambientRecording: true }), {
    moving: true, illuminated: false,
  })
  for (const lifecycle of ['waking', 'active']) {
    for (const ambientRecording of [false, true]) {
      assert.deepEqual(resolveOrbActivity({ lifecycle, ambientRecording }), {
        moving: true, illuminated: true,
      })
    }
  }
})

test('recording animates a sleeping orb independently of microphone input mute', () => {
  for (const enabled of [true, false]) {
    const className = desktopOrbClassName({ state: 'sleeping', lifecycle: 'sleeping', enabled, ambientRecording: true })
    assert.match(className, /\borb-moving\b/)
    assert.match(className, /\borb-dim\b/)
    assert.doesNotMatch(className, /\borb-lit\b/)
  }
})

test('returning to sleep keeps moving only while ambient recording is active', () => {
  const sleeping = { state: 'sleeping', lifecycle: 'sleeping', enabled: true }
  assert.match(desktopOrbClassName(sleeping), /\borb-still\b/)
  assert.match(desktopOrbClassName({ ...sleeping, ambientRecording: true }), /\borb-moving\b/)
  assert.match(desktopOrbClassName({ state: 'idle', lifecycle: 'active', enabled: false }), /\borb-moving orb-lit\b/)
})
