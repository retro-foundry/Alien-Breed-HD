import type { Mechanism } from '../model/level';

export function MechanismRulesInspector({ mechanism, prefix, change }: {
  mechanism: Pick<Mechanism, 'kind' | 'conditions' | 'openRule' | 'closeRule'>; prefix: string;
  change: (conditions: number, openRule: number, closeRule: number) => void;
}) {
  return <>
    <label className="select-field"><span>Opens when</span><select aria-label={`${prefix} opening rule`} title="Choose what starts opening or raising this mechanism. All selected key and switch requirements must also be met." value={mechanism.openRule}
      onChange={event => change(mechanism.conditions, Number(event.target.value), mechanism.closeRule)}>
      {(mechanism.kind === 'door' ? ['Player uses it', 'Player touches it', 'Bullet hits it', 'Enemy touches it', 'Timer expires', 'Never'] : ['Player uses it', 'Player touches it', 'Timer expires', 'Never']).map((label, value) => <option key={value} value={value}>{label}</option>)}
    </select></label>
    <label className="select-field"><span>Closes when</span><select aria-label={`${prefix} closing rule`} title="Choose what starts closing or lowering this mechanism." value={mechanism.closeRule}
      onChange={event => change(mechanism.conditions, mechanism.openRule, Number(event.target.value))}>
      {(mechanism.kind === 'door' ? ['Timer expires', 'Never'] : ['Player uses it', 'Player touches it', 'Timer expires', 'Never']).map((label, value) => <option key={value} value={value}>{label}</option>)}
    </select></label>
    <h4>Requires all selected</h4><div className="condition-grid">{Array.from({ length: 12 }, (_, bit) => <label className="check-field" key={bit}>
      <input type="checkbox" title="This requirement must be met along with every other checked key or switch in the level." checked={(mechanism.conditions & (1 << bit)) !== 0}
        onChange={event => change(event.target.checked ? mechanism.conditions | (1 << bit) : mechanism.conditions & ~(1 << bit), mechanism.openRule, mechanism.closeRule)} />
      {bit < 4 ? `Key ${bit + 1}` : `Switch ${bit - 3}`}
    </label>)}</div>
  </>;
}
