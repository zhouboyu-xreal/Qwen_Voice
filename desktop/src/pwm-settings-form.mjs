import { PWM_DEFAULTS } from '../../shared/pwm-model-settings.mjs'

export function createPwmSettingsForm({ document, desktop, translate: t, onChange }) {
  const controls = [...document.querySelectorAll('[data-pwm-field]')]
  const tests = [...document.querySelectorAll('[data-pwm-test]')]
  let unavailable = false
  let probing = false
  const values = () => Object.fromEntries(controls.map(control => [control.dataset.pwmField,
    control.type === 'checkbox' ? control.checked : control.type === 'number' ? Number(control.value) : control.value]))
  const prefix = kind => kind === 'llm' ? 'pwmLlm' : 'pwmEmbedding'
  const status = kind => document.querySelector(`[data-pwm-status="${kind}"]`)
  const preset = kind => document.querySelector(`[data-pwm-preset="${kind}"]`)
  const updatePreset = kind => {
    const draft = values()
    const key = prefix(kind)
    preset(kind).value = ['BaseUrl', 'Model'].every(suffix => draft[`${key}${suffix}`] === PWM_DEFAULTS[`${key}${suffix}`]) ? 'default' : 'custom'
  }
  const updateButtons = () => tests.forEach(button => { button.disabled = unavailable || probing })
  for (const control of controls) {
    const changed = () => {
      const kind = control.dataset.pwmField.startsWith('pwmLlm') ? 'llm' : 'embedding'
      updatePreset(kind)
      status(kind).textContent = ''
      onChange()
    }
    control.addEventListener('input', changed)
    control.addEventListener('change', changed)
  }
  for (const select of document.querySelectorAll('[data-pwm-preset]')) {
    select.addEventListener('change', () => {
      const kind = select.dataset.pwmPreset
      if (select.value === 'default') {
        for (const control of controls) {
          if (['BaseUrl', 'Model'].some(suffix => control.dataset.pwmField === `${prefix(kind)}${suffix}`)) {
            control.value = PWM_DEFAULTS[control.dataset.pwmField]
          }
        }
      }
      status(kind).textContent = ''
      onChange()
    })
  }
  for (const button of document.querySelectorAll('[data-pwm-reveal]')) {
    button.addEventListener('click', () => {
      const input = controls.find(control => control.dataset.pwmField === `${prefix(button.dataset.pwmReveal)}ApiKey`)
      const showing = input.type === 'password'
      input.type = showing ? 'text' : 'password'
      button.textContent = t(showing ? '隐藏' : '显示')
      button.setAttribute('aria-pressed', String(showing))
    })
  }
  for (const button of tests) {
    button.addEventListener('click', async () => {
      if (unavailable || probing) return
      const kind = button.dataset.pwmTest
      const output = status(kind)
      const draft = values()
      const snapshot = JSON.stringify(draft)
      probing = true
      updateButtons()
      output.className = 'pwm-test-status'
      output.textContent = t('正在测试连接…')
      try {
        const result = await desktop.testPwmModel(kind, { ...draft, gatewayUrl: document.querySelector('#gateway-url').value })
        if (snapshot !== JSON.stringify(values()) || unavailable) {
          output.textContent = t('参数已变更，请重新测试。')
          return
        }
        output.classList.add(result.ok ? 'success' : 'error')
        output.textContent = result.ok ? t('连接成功，模型响应正常。')
          : result.code === 'dimensions' ? t('服务返回 {dimensions} 维向量，请调整向量维度后重试。', { dimensions: result.dimensions })
            : result.code === 'http' ? t('连接失败（HTTP {status}），请检查地址、模型和密钥。', { status: result.status })
              : result.code === 'timeout' ? t('连接超时，请检查服务状态。')
                : result.code === 'key' ? t('请先填写 API Key。')
                  : t('连接失败，请检查服务地址、模型及返回格式。')
      } catch {
        output.classList.add('error')
        output.textContent = t('连接测试失败，请稍后重试。')
      } finally {
        probing = false
        updateButtons()
      }
    })
  }
  return {
    values,
    setUnavailable(value) { unavailable = value; updateButtons() },
    render(settings) {
      for (const control of controls) {
        const field = control.dataset.pwmField
        if (control.type === 'checkbox') control.checked = settings[field] ?? PWM_DEFAULTS[field]
        else control.value = settings[field] ?? PWM_DEFAULTS[field]
        if (field.endsWith('ApiKey')) {
          control.value = ''
          control.type = 'password'
          control.placeholder = t(settings[`${field}Configured`] ? '已配置；留空保留' : '输入 API Key')
        }
      }
      for (const button of document.querySelectorAll('[data-pwm-reveal]')) {
        button.textContent = t('显示'); button.setAttribute('aria-pressed', 'false')
      }
      for (const kind of ['llm', 'embedding']) { updatePreset(kind); status(kind).textContent = '' }
    },
  }
}
