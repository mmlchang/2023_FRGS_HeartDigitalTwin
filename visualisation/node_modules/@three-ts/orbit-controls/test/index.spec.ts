import * as Three from 'three';
import { OrbitControls } from '../src';
import { expect } from 'chai';
import { DOMWindow, JSDOM } from 'jsdom';

describe('orbit controls', () => {
  let controls: OrbitControls;
  let container: HTMLElement;
  let window: DOMWindow;

  beforeEach((done) => {
    const camera = new Three.PerspectiveCamera(50, 2, 1, 1000);
    const { window: domWindow } = new JSDOM(`<!DOCTYPE html><html><body><div id="container"></div></body></html>`);
    const { document } = domWindow;

    container = document.getElementById('container');
    controls = new OrbitControls(camera, container, domWindow as unknown as Window);
    window = domWindow;

    done();
  });

  afterEach(() => {
    window.close();
  });

  it('should be ok', () => {
    expect(controls).to.be.an.instanceOf(OrbitControls)
  });
});
